import pandas as pd
import os
import glob
import warnings

def load_and_preprocess_data(data_dir):
    """
    Loads all DAM_Market Snapshot Excel files from the directory, 
    combines them, and performs feature engineering.
    """
    all_files = glob.glob(os.path.join(data_dir, "DAM_Market Snapshot*.xlsx"))
    
    if not all_files:
        raise FileNotFoundError(f"No 'DAM_Market Snapshot' files found in {data_dir}")
        
    df_list = []
    
    print(f"Found {len(all_files)} files.")
    
    for file in all_files:
        try:
            # Read first few rows to find header
            # We assume header contains 'Date' or 'Time Block'
            # Reading a chunk to find the header row index
            temp_df = pd.read_excel(file, header=None, nrows=20)
            header_row_idx = None
            for i, row in temp_df.iterrows():
                row_str = row.astype(str).str.lower().tolist()
                # Loosen check: just 'date' and 'mcp' or 'mcv'
                if any('date' in s for s in row_str) and (any('mcp' in s for s in row_str) or any('mcv' in s for s in row_str)):
                    header_row_idx = i
                    print(f"Header found at row {i} in {os.path.basename(file)}")
                    break
            
            if header_row_idx is None:
                print(f"Skipping {os.path.basename(file)}: Could not find header. First row: {temp_df.iloc[0].tolist()}")
                continue
                
            # Read actual data
            df = pd.read_excel(file, header=header_row_idx)
            
            # Standardize columns
            df.columns = [str(c).strip() for c in df.columns]
            
            df_list.append(df)
            
        except Exception as e:
            print(f"Error loading {os.path.basename(file)}: {e}")

    if not df_list:
        raise ValueError("No data could be loaded.")

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Raw stacked shape: {full_df.shape}")
    
    # --- Preprocessing ---
    
    # 1. Date Parsing
    date_col = next((c for c in full_df.columns if 'date' in c.lower()), None)
    mcp_col = next((c for c in full_df.columns if 'mcp' in c.lower()), None)
    
    if not date_col:
        print("Columns found:", full_df.columns.tolist())
        raise ValueError("Date column not found")
        
    # debug date before conversion
    print(f"Sample raw dates ({date_col}):", full_df[date_col].head(5).tolist())

    full_df['Date'] = pd.to_datetime(full_df[date_col], errors='coerce', dayfirst=True)
    
    # Check NaT
    nat_count = full_df['Date'].isna().sum()
    if nat_count > 0:
        print(f"Dropped {nat_count} rows due to invalid Date parsing.")
    
    full_df = full_df.dropna(subset=['Date'])
    
    # 2. Sort
    tb_col = next((c for c in full_df.columns if 'block' in c.lower()), 'Time Block')
    
    # Check if TimeBlock is string and needs parsing
    # Sample value check
    sample_val = full_df[tb_col].dropna().iloc[0]
    print(f"Sample TimeBlock value: {sample_val} (Type: {type(sample_val)})")
    
    def parse_time_block(val):
        # Handle "00:00 - 00:15" format
        if isinstance(val, str) and '-' in val:
            try:
                start_time = val.split('-')[0].strip()
                hh, mm = map(int, start_time.split(':'))
                return hh * 4 + (mm // 15) + 1
            except:
                return float('nan')
        # Handle numeric 1-96
        if isinstance(val, (int, float)):
            return int(val)
        return float('nan')

    full_df['TimeBlockInt'] = full_df[tb_col].apply(parse_time_block)
    
    # Drop rows where TimeBlockInt is NaN
    full_df = full_df.dropna(subset=['TimeBlockInt'])
    full_df['TimeBlock'] = full_df['TimeBlockInt'].astype(int)
    
    # Sort
    full_df = full_df.sort_values(by=['Date', 'TimeBlock'])

    # 3. Numeric Conversions
    cols_to_numeric = [mcp_col]
    # Add other feature columns if they exist
    purchase_col = next((c for c in full_df.columns if 'purchase' in c.lower()), None)
    sell_col = next((c for c in full_df.columns if 'sell' in c.lower()), None)
    
    if purchase_col: cols_to_numeric.append(purchase_col)
    if sell_col: cols_to_numeric.append(sell_col)
        
    for col in cols_to_numeric:
        # Avoid coercing the already converted TimeBlock
        if col == tb_col: continue
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # Rename for consistency
    # Note: we already have 'TimeBlock' (int) and 'Date'.
    # We need to rename MCP etc.
    rename_map = {mcp_col: 'MCP'}
    if purchase_col: rename_map[purchase_col] = 'PurchaseBid'
    if sell_col: rename_map[sell_col] = 'SellBid'
    
    full_df = full_df.rename(columns=rename_map)
    
    print(f"Combined Shape after cleanup: {full_df.shape}")
    print("Columns:", full_df.columns.tolist())
    
    # Handle duplicates
    full_df = full_df.drop_duplicates(subset=['Date', 'TimeBlock'], keep='last')

    # --- Feature Engineering ---
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning) # formatting
    
    df = full_df.copy()
    
    # Lag Features (Past values)
    # Since we predict Day-Ahead, we must use data available at D-1.
    target = 'MCP'
    
    # Sort to be sure shift works temporally
    df = df.sort_values(by=['Date', 'TimeBlock'])
    
    df['MCP_Lag_24h'] = df[target].shift(96)
    df['MCP_Lag_48h'] = df[target].shift(192)
    df['MCP_Lag_168h'] = df[target].shift(96*7)
    
    print(f"Shape before dropna (lags): {df.shape}")
    
    # Rolling stats
    df['MCP_Rolling_Mean_24h'] = df['MCP_Lag_24h'].rolling(window=96).mean()
    df['MCP_Rolling_Std_24h'] = df['MCP_Lag_24h'].rolling(window=96).std()
    
    # Time Features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    import numpy as np
    df['TimeBlock_Sin'] = np.sin(2 * np.pi * df['TimeBlock'] / 96)
    df['TimeBlock_Cos'] = np.cos(2 * np.pi * df['TimeBlock'] / 96)
    
    # Drop NaNs created by lags
    df_clean = df.dropna().reset_index(drop=True)
    print(f"Shape after dropna: {df_clean.shape}")
    
    if df_clean.empty:
        print("WARNING: Dataframe is empty after dropping NaNs!")
        print("NaN counts:\n", df.isna().sum())
    
    return df_clean



if __name__ == "__main__":
    # Test run
    try:
        df = load_and_preprocess_data("d:/shubham/dam for/")
        print(df.head())
        print(df.tail())
    except Exception as e:
        print(e)
