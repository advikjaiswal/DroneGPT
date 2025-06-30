# actions.py
import pandas as pd

# === Load full feature CSV ===
try:
    df = pd.read_csv("drone_data/full.csv")
except Exception as e:
    raise RuntimeError(f"Failed to load drone_data/full.csv: {e}")

if "pitchspeed" not in df.columns:
    raise RuntimeError("Column 'pitchspeed' not found in drone_data/full.csv")

vx_values = df["pitchspeed"].values

# === Label actions based on a smarter threshold ===
# Set 0.05 as threshold to detect subtle movement
labels = ["Hover" if abs(vx) <= 0.05 else "Move" for vx in vx_values]

# === Write vx|action to training text file ===
with open("gpt_action_training.txt", "w") as f:
    for vx, label in zip(vx_values, labels):
        f.write(f"{vx:.3f}|{label} ")

print("✅ gpt_action_training.txt generated.")
