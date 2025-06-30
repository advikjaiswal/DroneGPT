import pandas as pd
df = pd.read_csv("drone_data/3ddb3350-469e-4508-a887-59b5531f4487_vehicle_local_position_0.csv")
vx = df["vx"].dropna()
vx.to_csv("drone_data/vx.csv", index=False)
print("✅ vx.csv saved to drone_data/")
