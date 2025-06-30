# generate_full_csv.py

import pandas as pd
import os

# Folder where CSVs are stored
DATA_DIR = "drone_data"
ULG_PREFIX = None

# Automatically detect the ulg UUID prefix
for file in os.listdir(DATA_DIR):
    if file.endswith("_vehicle_imu_0.csv"):
        ULG_PREFIX = file.split("_vehicle_imu_0.csv")[0]
        break

if not ULG_PREFIX:
    raise FileNotFoundError("❌ Could not find a valid ULG prefix from drone_data/*.csv")

print(f"✅ Using ULG prefix: {ULG_PREFIX}")

# Define the 16 features and their source CSV and column
FEATURE_SPECS = {
    "vx": ("vehicle_local_position_0", "vx"),
    "vy": ("vehicle_local_position_0", "vy"),
    "vz": ("vehicle_local_position_0", "vz"),
    "rollspeed": ("vehicle_angular_velocity_0", "xyz[0]"),
    "pitchspeed": ("vehicle_angular_velocity_0", "xyz[1]"),
    "yawspeed": ("vehicle_angular_velocity_0", "xyz[2]"),
    "ax": ("vehicle_imu_0", "delta_velocity[0]"),
    "ay": ("vehicle_imu_0", "delta_velocity[1]"),
    "az": ("vehicle_imu_0", "delta_velocity[2]"),
    "rollacc": ("vehicle_imu_0", "delta_angle[0]"),
    "pitchacc": ("vehicle_imu_0", "delta_angle[1]"),
    "yawacc": ("vehicle_imu_0", "delta_angle[2]"),
    "q1": ("vehicle_attitude_0", "q[0]"),
    "q2": ("vehicle_attitude_0", "q[1]"),
    "q3": ("vehicle_attitude_0", "q[2]"),
    "q4": ("vehicle_attitude_0", "q[3]"),
}

dfs = []

for feature, (topic, column) in FEATURE_SPECS.items():
    file_path = os.path.join(DATA_DIR, f"{ULG_PREFIX}_{topic}.csv")

    if not os.path.exists(file_path):
        print(f"❌ Missing topic CSV: {file_path} — skipping {feature}")
        continue

    df = pd.read_csv(file_path)

    if column not in df.columns:
        print(f"⚠️ Column '{column}' not found in {file_path}")
        continue

    feature_df = df[[column]].rename(columns={column: feature})
    dfs.append(feature_df)

# Check if all 16 features were loaded
if len(dfs) != len(FEATURE_SPECS):
    print(f"❌ Only {len(dfs)} out of {len(FEATURE_SPECS)} features found. Cannot continue.")
    exit(1)

# Concatenate all columns side by side
final_df = pd.concat(dfs, axis=1)

# Save to CSV
output_path = os.path.join(DATA_DIR, "full.csv")
final_df.to_csv(output_path, index=False)
print(f"✅ Saved combined feature file to {output_path}")
