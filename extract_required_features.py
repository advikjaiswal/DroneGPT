# extract_required_features.py

import os
import glob
import pandas as pd

# Map: output name → (wildcard match, column)
feature_sources = {
    "vx": ("*vehicle_local_position_0.csv", "vx"),
    "vy": ("*vehicle_local_position_0.csv", "vy"),
    "vz": ("*vehicle_local_position_0.csv", "vz"),
    "rollspeed": ("*vehicle_angular_velocity_0.csv", "xyz[0]"),
    "pitchspeed": ("*vehicle_angular_velocity_0.csv", "xyz[1]"),
    "yawspeed": ("*vehicle_angular_velocity_0.csv", "xyz[2]"),
    "ax": ("*vehicle_imu_0.csv", "accelerometer_m_s2[0]"),
    "ay": ("*vehicle_imu_0.csv", "accelerometer_m_s2[1]"),
    "az": ("*vehicle_imu_0.csv", "accelerometer_m_s2[2]"),
    "rollacc": ("*vehicle_imu_0.csv", "gyros_rad[0]"),
    "pitchacc": ("*vehicle_imu_0.csv", "gyros_rad[1]"),
    "yawacc": ("*vehicle_imu_0.csv", "gyros_rad[2]"),
    "q1": ("*vehicle_attitude_0.csv", "q[0]"),
    "q2": ("*vehicle_attitude_0.csv", "q[1]"),
    "q3": ("*vehicle_attitude_0.csv", "q[2]"),
    "q4": ("*vehicle_attitude_0.csv", "q[3]"),
}

os.makedirs("drone_data", exist_ok=True)

for name, (pattern, col) in feature_sources.items():
    matches = glob.glob(f"drone_data/{pattern}")
    if not matches:
        print(f"❌ Missing topic CSV for {name} → pattern `{pattern}`")
        continue

    file_path = matches[0]  # Use the first match
    try:
        df = pd.read_csv(file_path)
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found in {file_path}")
            continue
        series = df[col].dropna().astype("float32")
        series.to_csv(f"drone_data/{name}.csv", index=False)
        print(f"✅ Saved {name}.csv from {os.path.basename(file_path)}")
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
