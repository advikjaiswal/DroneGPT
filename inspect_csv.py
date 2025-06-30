import pandas as pd
import os

folder = "drone_data"
keywords = ["local_position", "airspeed", "velocity"]

for file in os.listdir(folder):
    if any(k in file for k in keywords):
        try:
            path = os.path.join(folder, file)
            df = pd.read_csv(path, nrows=1)
            print(f"\n{file} -> {list(df.columns)}")
        except Exception as e:
            print(f"Failed to read {file}: {e}")
