import os
import pandas as pd

INPUT_DIR = "drone_data"
OUTPUT_FILE = "data/drone_logs.txt"

os.makedirs("data", exist_ok=True)
with open(OUTPUT_FILE, "w") as out:
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(INPUT_DIR, filename)
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                line = ','.join(map(str, row.values))
                out.write(line + "\n")

print(f"✅ Saved merged log lines to {OUTPUT_FILE}")
