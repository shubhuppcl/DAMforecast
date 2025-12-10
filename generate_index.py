import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.io as pio
from data_loader import load_and_preprocess_data
import os
import sys

# Ensure output directory exists (for local testing)
OUT_DIR = "public"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def generate_static_dashboard():
    print("Starting Static Dashboard Generation...")
    
    # 1. Load Data
    try:
        data_dir = "./" # Assumes running from root of repo where files are
        # If running locally in 'd:/shubham/dam for', adjust or assume files are there
        if os.path.exists("d:/shubham/dam for/"):
            data_dir = "d:/shubham/dam for/"
            
        print(f"Loading data from {data_dir}...")
        df = load_and_preprocess_data(data_dir)
        print(f"Data Loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 2. Determine Prediction Target (Latest Date)
    latest_date = df['Date'].max()
    print(f"Target Date: {latest_date.date()}")
    
    # 3. Prepare Data
    features = [
        'MCP_Lag_24h', 'MCP_Lag_48h', 'MCP_Lag_168h', 
        'MCP_Rolling_Mean_24h', 'MCP_Rolling_Std_24h',
        'TimeBlock', 'DayOfWeek', 'Month', 
        'TimeBlock_Sin', 'TimeBlock_Cos'
    ]
    # Check extra features
    if 'PurchaseBid' in df.columns and df['PurchaseBid'].notna().mean() > 0.9:
        features.append('PurchaseBid')
    if 'SellBid' in df.columns and df['SellBid'].notna().mean() > 0.9:
        features.append('SellBid')
        
    target = 'MCP'
    
    train_df = df[df['Date'] < latest_date]
    test_df = df[df['Date'] == latest_date]
    
    if train_df.empty or test_df.empty:
        print("Error: insufficient data for training/testing")
        sys.exit(1)
        
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    
    # 4. Train Model
    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        objective='reg:squarederror',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 5. Predict & Clip
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 10000)
    
    # 6. Calc Metrics
    actuals = test_df['MCP'].values
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    mae = np.mean(np.abs(actuals - preds))
    avg_price = np.mean(actuals)
    
    # 7. Generate Plotly Figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df['TimeBlock'], y=actuals, mode='lines+markers', name='Actual MCP'))
    fig.add_trace(go.Scatter(x=test_df['TimeBlock'], y=preds, mode='lines+markers', name='Predicted MCP', line=dict(dash='dash')))
    
    title_text = f"DAM Price Forecast for {latest_date.strftime('%d %b %Y')}<br><sup>MAPE: {mape:.2f}% | MAE: {mae:.2f}</sup>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Time Block",
        yaxis_title="Price (MCP)",
        template="plotly_white",
        height=600
    )
    
    # 8. Create HTML Content
    # We will embed the plotly graph in a nice HTML wrapper
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DAM Price Forecast</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f9; }}
            .container {{ max_width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; text-align: center; }}
            .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; background: #eee; padding: 15px; border-radius: 5px; }}
            .metric {{ text-align: center; }}
            .metric-val {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
            .footer {{ text-align: center; margin-top: 30px; color: #777; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚡ DAM Price Forecast</h1>
            <p style="text-align:center">Prediction for <strong>{latest_date.strftime('%Y-%m-%d')}</strong></p>
            
            <div class="metrics">
                <div class="metric">MAPE<br><span class="metric-val">{mape:.2f}%</span></div>
                <div class="metric">MAE<br><span class="metric-val">{mae:.2f}</span></div>
                <div class="metric">Avg Price<br><span class="metric-val">{avg_price:.2f}</span></div>
            </div>
            
            {plot_html}
            
            <div class="footer">
                Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: XGBoost
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Dashboard generated at: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    generate_static_dashboard()
