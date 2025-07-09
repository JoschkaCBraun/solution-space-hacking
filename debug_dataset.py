#!/usr/bin/env python3
"""Debug dataset loading issue."""

import sys
print("Starting debug script...", flush=True)

print("\nStep 1: Importing pandas...", flush=True)
import pandas as pd
print("  Pandas imported successfully", flush=True)

print("\nStep 2: Loading parquet file...", flush=True)
file_path = "data/APPS/cleaned/apps_eval.parquet"
try:
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df)} rows", flush=True)
    print(f"  Columns: {list(df.columns)}", flush=True)
except Exception as e:
    print(f"  Error: {e}", flush=True)
    sys.exit(1)

print("\nStep 3: Converting to dict...", flush=True)
try:
    # Just try first row
    first_row = df.iloc[0]
    row_dict = first_row.to_dict()
    print(f"  First row problem_id: {row_dict.get('problem_id')}", flush=True)
    print(f"  Keys: {list(row_dict.keys())[:5]}...", flush=True)
except Exception as e:
    print(f"  Error: {e}", flush=True)

print("\nStep 4: Iterating through rows...", flush=True)
try:
    count = 0
    for idx, row in df.iterrows():
        count += 1
        if count > 5:
            break
        print(f"  Row {count}: problem_id={row['problem_id']}", flush=True)
except Exception as e:
    print(f"  Error during iteration: {e}", flush=True)

print("\nDebug complete!", flush=True)