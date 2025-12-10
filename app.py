import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from data_loader import load_and_preprocess_data
import os

# Set page config
st.set_page_config(page_title="DAM Price Predictor", layout="wide")

@st.cache_data
def get_data():
    return load_and_preprocess_data("d:/shubham/dam for/")

def train_and_predict(df, target_date):
    # Features
    features = [
        'MCP_Lag_24h', 'MCP_Lag_48h', 'MCP_Lag_168h', 
        'MCP_Rolling_Mean_24h', 'MCP_Rolling_Std_24h',
        'TimeBlock', 'DayOfWeek', 'Month', 
        'TimeBlock_Sin', 'TimeBlock_Cos'
    ]
    
    # Check if 'PurchaseBid' and 'SellBid' are valid (not mostly nan)
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
    y_test = test_df[target]
    
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
    
    # 1. Price constraints: [0, 10000]
    preds = np.clip(preds, 0, 10000)
    
    return test_df, preds, model, features


def main():
    st.title("⚡ DAM Price Predictor & Dashboard")
    st.markdown("Automated XGBoost prediction for Day-Ahead Market prices.")
    
    with st.spinner("Loading and processing data from Excel files..."):
        try:
            df = get_data()
            st.success(f"Loaded {len(df)} rows. Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    # Identify Target Date (Latest available)
    latest_date = df['Date'].max()
    st.header(f"Prediction for Target Date: {latest_date.date()}")
    
    # Run Model
    with st.spinner("Training XGBoost model and generating predictions..."):
        test_df, preds, model, feature_names = train_and_predict(df, latest_date)
        
    if test_df is None:
        st.error("Model training failed.")
        return
        
    # Metrics
    actuals = test_df['MCP'].values
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    mae = np.mean(np.abs(actuals - preds))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAPE", f"{mape:.2f}%")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("Avg Price", f"{np.mean(actuals):.2f}")
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df['TimeBlock'], y=actuals, mode='lines+markers', name='Actual MCP'))
    fig.add_trace(go.Scatter(x=test_df['TimeBlock'], y=preds, mode='lines+markers', name='Predicted MCP', line=dict(dash='dash')))
    
    fig.update_layout(
        title=f"MCP Forecast vs Actual for {latest_date.date()}",
        xaxis_title="Time Block",
        yaxis_title="Price (MCP)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    st.subheader("Detailed Data")
    results = test_df[['Date', 'TimeBlock', 'MCP']].copy()
    results['Predicted_MCP'] = preds
    results['Diff'] = results['MCP'] - results['Predicted_MCP']
    results['Error_%'] = (results['Diff'].abs() / results['MCP']) * 100
    
    st.dataframe(results.style.format({
        'MCP': '{:.2f}', 'Predicted_MCP': '{:.2f}', 
        'Diff': '{:.2f}', 'Error_%': '{:.2f}'
    }))
    
    # Feature Importance
    if st.checkbox("Show Feature Importance"):
        imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(imp.set_index('Feature'))

if __name__ == "__main__":
    main()
