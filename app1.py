import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from data_loader import load_and_preprocess_data
import os
from datetime import timedelta

# Set page config
st.set_page_config(page_title="DAM Price Predictor (Extended)", layout="wide")

@st.cache_data
def get_data():
    return load_and_preprocess_data("d:/shubham/dam for/")

def train_and_predict(df, target_date, use_bids=True):
    """
    Trains on data < target_date and predicts for target_date.
    use_bids: If False, exclude PurchaseBid and SellBid from features.
    """
    # Features
    features = [
        'MCP_Lag_24h', 'MCP_Lag_48h', 'MCP_Lag_168h', 
        'MCP_Rolling_Mean_24h', 'MCP_Rolling_Std_24h',
        'TimeBlock', 'DayOfWeek', 'Month', 
        'TimeBlock_Sin', 'TimeBlock_Cos'
    ]
    
    # Check if 'PurchaseBid' and 'SellBid' are valid (not mostly nan)
    # AND we are allowed to use them
    if use_bids:
        if 'PurchaseBid' in df.columns and df['PurchaseBid'].notna().mean() > 0.9:
            features.append('PurchaseBid')
        if 'SellBid' in df.columns and df['SellBid'].notna().mean() > 0.9:
            features.append('SellBid')
        
    target = 'MCP'
    
    # Split
    # Train: < target_date
    # Test: == target_date
    train_df = df[df['Date'] < target_date]
    test_df = df[df['Date'] == target_date]
    
    if train_df.empty or test_df.empty:
        return None, None, None, f"Insufficient data. Train: {len(train_df)}, Test: {len(test_df)}"
        
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Predict
    preds = model.predict(X_test)
    
    # Price constraints: [0, 10000]
    preds = np.clip(preds, 0, 10000)
    
    return test_df, preds, model, features

def prepare_future_dataframe(base_df):
    """
    Creates a DataFrame for the Next Day (D+1) by appending empty rows
    and populating Lag features from base_df.
    """
    latest_date = base_df['Date'].max()
    next_date = latest_date + timedelta(days=1)
    
    # Create 96 blocks for next_date
    future_data = []
    for tb in range(1, 97):
        future_data.append({
            'Date': next_date,
            'TimeBlock': tb,
            'MCP': np.nan # Unknown
        })
    
    future_df = pd.DataFrame(future_data)
    
    # Combine with history to calculate lags
    # We only need enough history to calculate max lag (1 week = 7days)
    # But simpler to just concat provided base_df
    extended_df = pd.concat([base_df, future_df], ignore_index=True)
    extended_df = extended_df.sort_values(by=['Date', 'TimeBlock'])
    
    # Re-calculate Lags
    # Note: data_loader already did this, but we need to do it for the new rows
    # We must replicate the logic exactly
    target = 'MCP'
    df = extended_df.copy() # working copy
    
    # Shift
    df['MCP_Lag_24h'] = df[target].shift(96)
    df['MCP_Lag_48h'] = df[target].shift(192)
    df['MCP_Lag_168h'] = df[target].shift(96*7)
    
    # Rolling (on Lag 24h)
    df['MCP_Rolling_Mean_24h'] = df['MCP_Lag_24h'].rolling(window=96).mean()
    df['MCP_Rolling_Std_24h'] = df['MCP_Lag_24h'].rolling(window=96).std()
    
    # Time Features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['TimeBlock_Sin'] = np.sin(2 * np.pi * df['TimeBlock'] / 96)
    df['TimeBlock_Cos'] = np.cos(2 * np.pi * df['TimeBlock'] / 96)
    
    # Return ONLY the future rows
    return df[df['Date'] == next_date].copy(), df

def main():
    st.title("⚡ DAM Price Predictor (Extended)")
    st.markdown("Automated XGBoost prediction for Day-Ahead Market prices.")
    
    with st.spinner("Loading and processing data..."):
        try:
            df = get_data()
            st.success(f"Data Loaded. Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    # --- Section 1: Validation (Latest Actual Date) ---
    latest_date = df['Date'].max()
    st.markdown("---")
    st.header(f"1. Validation: {latest_date.date()} (Actual vs Predicted)")
    
    with st.spinner("Training Model 1 (Validation)..."):
        # Use Bids if available for validation to match 'app.py' logic
        test_df, preds, model, features = train_and_predict(df, latest_date, use_bids=True)
        
    if test_df is not None:
        # Metrics
        actuals = test_df['MCP'].values
        mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
        mae = np.mean(np.abs(actuals - preds))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE", f"{mape:.2f}%")
        c2.metric("MAE", f"{mae:.2f}")
        c3.metric("Avg Price", f"{np.mean(actuals):.2f}")
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_df['TimeBlock'], y=actuals, mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=test_df['TimeBlock'], y=preds, mode='lines+markers', name='Predicted', line=dict(dash='dash')))
        fig.update_layout(title=f"Forecast vs Actual ({latest_date.date()})", xaxis_title="Block", yaxis_title="MCP")
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Section 2: Future Forecast (Next Day) ---
    next_date = latest_date + timedelta(days=1)
    st.markdown("---")
    st.header(f"2. Future Forecast: {next_date.date()} (Next Day)")
    st.caption("Training separate model WITHOUT Bid data (Pure Autoregressive) for future prediction.")

    with st.spinner("Generating inputs and Training Model 2..."):
        # 1. Prepare Future Feature Set
        future_rows, full_extended_df = prepare_future_dataframe(df)
        
        # 2. Train Model WITHOUT Bids (since we don't have them for tomorrow)
        # We train on ALL history up to latest_date
        # We pass full_extended_df but split < next_date
        # train_and_predict splits by < target_date.
        # So we pass target_date = next_date.
        future_test_df, future_preds, future_model, future_feats = train_and_predict(
            full_extended_df, 
            next_date, 
            use_bids=False # Important!
        )
        
    if future_test_df is not None:
        # Metrics (No Actuals available)
        avg_pred = np.mean(future_preds)
        st.metric("Predicted Avg Price", f"{avg_pred:.2f}")
        
        # Plot
        fig2 = go.Figure()
        # No actuals to plot
        fig2.add_trace(go.Scatter(x=future_test_df['TimeBlock'], y=future_preds, mode='lines+markers', name='Forecast', line=dict(color='orange')))
        fig2.update_layout(title=f"Price Forecast for {next_date.date()}", xaxis_title="Block", yaxis_title="Predicted MCP")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Table
        st.subheader("Forecast Details")
        res = pd.DataFrame({
            'TimeBlock': future_test_df['TimeBlock'],
            'Predicted_MCP': future_preds
        })
        st.dataframe(res.style.format({'Predicted_MCP': '{:.2f}'}))

if __name__ == "__main__":
    main()
