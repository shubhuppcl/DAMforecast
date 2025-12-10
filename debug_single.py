import pandas as pd
import os
import glob

# Try loading ONE file and inspect strictly
file_path = "d:/shubham/dam for/DAM_Market Snapshot Dec 2025.xlsx"

print(f"Checking {file_path}")
# Read 20 rows
df_raw = pd.read_excel(file_path, header=None, nrows=20)
print("--- Raw 20 rows ---")
print(df_raw)

# Identify header
header_idx = -1
for i, row in df_raw.iterrows():
    s = row.astype(str).str.lower().tolist()
    if 'date' in s and ('mcp' in str(s) or 'mcv' in str(s)):
        header_idx = i
        print(f"Header candidates found at row {i}")
        break

if header_idx != -1:
    df = pd.read_excel(file_path, header=header_idx, nrows=20)
    print("--- Loaded with Header ---")
    print(df.columns.tolist())
    print(df.head())
    
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    print(f"Date Col: {date_col}")
    if date_col:
        print("Date Values Sample:")
        print(df[date_col].head())
        
        print("Attempting conversion:")
        converted = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        print(converted.head())
        print("NaT count in sample:", converted.isna().sum())
