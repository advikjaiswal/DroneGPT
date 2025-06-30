import os
import pandas as pd

DATA_DIR = "drone_data"

print("\n🔍 Checking files for 'velocity[0]' column:\n")
for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue
    path = os.path.join(DATA_DIR, file)
    try:
        df = pd.read_csv(path, nrows=1)
        cols = list(df.columns)
        if "velocity[0]" in cols:
            print(f"✅ {file} contains 'velocity[0]'")
        else:
            print(f"❌ {file} does NOT contain 'velocity[0]' — Columns: {cols}")
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")
