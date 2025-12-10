import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.io as pio
from data_loader import load_and_preprocess_data
import os
import sys
from datetime import timedelta

# Ensure output directory exists
OUT_DIR = "public"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def train_and_predict_logic(df, target_date, use_bids=True):
    # Features
    features = [
        'MCP_Lag_24h', 'MCP_Lag_48h', 'MCP_Lag_168h', 
        'MCP_Rolling_Mean_24h', 'MCP_Rolling_Std_24h',
        'TimeBlock', 'DayOfWeek', 'Month', 
        'TimeBlock_Sin', 'TimeBlock_Cos'
    ]
    if use_bids:
        if 'PurchaseBid' in df.columns and df['PurchaseBid'].notna().mean() > 0.9:
            features.append('PurchaseBid')
        if 'SellBid' in df.columns and df['SellBid'].notna().mean() > 0.9:
            features.append('SellBid')
        
    target = 'MCP'
    train_df = df[df['Date'] < target_date]
    test_df = df[df['Date'] == target_date]
    
    if train_df.empty:
        return None, None, None
        
    X_train = train_df[features]
    y_train = train_df[target]
    
    # Train
    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6, 
        subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict
    if test_df.empty:
        return None, None, model # Only return model if no test data
        
    X_test = test_df[features]
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 10000)
    
    return test_df, preds, model

def prepare_future_dataframe(base_df):
    latest_date = base_df['Date'].max()
    next_date = latest_date + timedelta(days=1)
    
    future_data = []
    for tb in range(1, 97):
        future_data.append({'Date': next_date, 'TimeBlock': tb, 'MCP': np.nan})
    
    future_df = pd.DataFrame(future_data)
    extended_df = pd.concat([base_df, future_df], ignore_index=True)
    extended_df = extended_df.sort_values(by=['Date', 'TimeBlock'])
    
    # Lags
    target = 'MCP'
    df = extended_df.copy()
    df['MCP_Lag_24h'] = df[target].shift(96)
    df['MCP_Lag_48h'] = df[target].shift(192)
    df['MCP_Lag_168h'] = df[target].shift(96*7)
    
    df['MCP_Rolling_Mean_24h'] = df['MCP_Lag_24h'].rolling(window=96).mean()
    df['MCP_Rolling_Std_24h'] = df['MCP_Lag_24h'].rolling(window=96).std()
    
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['TimeBlock_Sin'] = np.sin(2 * np.pi * df['TimeBlock'] / 96)
    df['TimeBlock_Cos'] = np.cos(2 * np.pi * df['TimeBlock'] / 96)
    
    return df[df['Date'] == next_date].copy(), df

def generate_static_dashboard():
    print("Starting Static Dashboard Generation...")
    
    try:
        data_dir = "./"
        if os.path.exists("d:/shubham/dam for/"):
            data_dir = "d:/shubham/dam for/"
            
        print(f"Loading data from {os.path.abspath(data_dir)}...")
        # Check files
        import glob
        files = glob.glob(os.path.join(data_dir, "DAM_Market Snapshot*.xlsx"))
        print(f"Files found: {len(files)}")
        
        df = load_and_preprocess_data(data_dir)
        print(f"Data Loaded. Shape: {df.shape}")

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # --- 1. Validation (Latest Date) ---
    latest_date = df['Date'].max()
    print(f"Validation Target Date: {latest_date.date()}")
    
    test_df, preds, _ = train_and_predict_logic(df, latest_date, use_bids=True)
    
    fig_val_html = ""
    metrics_html = ""
    
    if test_df is not None:
        actuals = test_df['MCP'].values
        mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
        mae = np.mean(np.abs(actuals - preds))
        avg_price = np.mean(actuals)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_df['TimeBlock'], y=actuals, mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=test_df['TimeBlock'], y=preds, mode='lines+markers', name='Predicted', line=dict(dash='dash')))
        fig.update_layout(title="", xaxis_title="Block", yaxis_title="MCP", template="plotly_white", height=500)
        fig_val_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
        metrics_html = f"""
            <div class="metrics">
                <div class="metric">MAPE<br><span class="metric-val">{mape:.2f}%</span></div>
                <div class="metric">MAE<br><span class="metric-val">{mae:.2f}</span></div>
                <div class="metric">Avg Price<br><span class="metric-val">{avg_price:.2f}</span></div>
            </div>
        """
    
    # --- 2. Future Forecast (Next Day) ---
    next_date = latest_date + timedelta(days=1)
    print(f"Future Target Date: {next_date.date()}")
    
    future_rows, full_extended_df = prepare_future_dataframe(df)
    future_test_df, future_preds, _ = train_and_predict_logic(full_extended_df, next_date, use_bids=False)
    
    fig_fut_html = ""
    fut_metrics_html = ""
    
    if future_preds is not None:
        avg_pred = np.mean(future_preds)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=future_test_df['TimeBlock'], y=future_preds, mode='lines+markers', name='Forecast', line=dict(color='orange')))
        fig2.update_layout(title="", xaxis_title="Block", yaxis_title="MCP", template="plotly_white", height=500)
        fig_fut_html = pio.to_html(fig2, full_html=False, include_plotlyjs=False) # JS already included above
        
        fut_metrics_html = f"""
            <div class="metrics" style="background: #fff3cd;">
                <div class="metric">Predicted Avg Price<br><span class="metric-val" style="color: #856404;">{avg_pred:.2f}</span></div>
            </div>
        """

    # --- HTML Assembly ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DAM Price Forecast</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background-color: #f8f9fa; }}
            .container {{ max_width: 1100px; margin: 30px auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 5px; }}
            h2 {{ color: #495057; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 40px; }}
            .subtitle {{ text-align: center; color: #6c757d; margin-bottom: 30px; }}
            .metrics {{ display: flex; justify-content: center; gap: 40px; margin: 20px 0; background: #e9ecef; padding: 20px; border-radius: 8px; }}
            .metric {{ text-align: center; }}
            .metric-val {{ font-size: 1.8em; font-weight: bold; color: #007bff; display: block; margin-top: 5px; }}
            .footer {{ text-align: center; margin-top: 50px; color: #adb5bd; font-size: 0.8em; border-top: 1px solid #eee; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚡ DAM Price Forecast Dashboard</h1>
            <p class="subtitle">Automated AI Prediction | Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <!-- Future Section First (Most Important) -->
            <h2>🚀 Future Forecast: {next_date.strftime('%d %b %Y')}</h2>
            {fut_metrics_html}
            {fig_fut_html}
            
            <!-- Validation Section -->
            <h2>✅ Model Validation: {latest_date.strftime('%d %b %Y')}</h2>
            <p>Verification against published actuals.</p>
            {metrics_html}
            {fig_val_html}
            
            <div class="footer">
                Model: XGBoost (Autoregressive) | Data Source: IEX DAM Market Snapshots
            </div>
        </div>
    </body>
    </html>
    """
    
    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Dashboard generated at: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    generate_static_dashboard()

