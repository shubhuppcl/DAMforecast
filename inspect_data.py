import pandas as pd
import os


files_to_check = [
    "d:/shubham/dam for/DAM_Market Snapshot Dec 2025.xlsx",
    "d:/shubham/dam for/DAM_Market Snapshot Apr 2023.xlsx"
]

for file_path in files_to_check:
    print(f"\n--- Checking {os.path.basename(file_path)} ---")
    try:
        # Based on previous output, header likely around row 5 (0-indexed) or 6 (1-indexed)
        # Row 9 had data. Let's look closer at rows 4-8.
        df = pd.read_excel(file_path, header=None)
        
        print("Rows 4 to 9:")
        for i in range(4, 10):
            print(f"Row {i}: {df.iloc[i].tolist()}")
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

