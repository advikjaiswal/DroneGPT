import pandas as pd
import numpy as np
import os

# Load the file with real velocity data
df = pd.read_csv("drone_data/3ddb3350-469e-4508-a887-59b5531f4487_vehicle_local_position_0.csv")

# Use vx (forward velocity)
data = df["vx"].dropna().values

# Normalize between 0 and 1 for stability
data_min = data.min()
data_max = data.max()
norm_data = (data - data_min) / (data_max - data_min)

# Prepare sequences for prediction
block_size = 16
X = []
y = []
for i in range(len(norm_data) - block_size):
    X.append(norm_data[i:i+block_size])
    y.append(norm_data[i+block_size])

X = np.array(X)
y = np.array(y)

# Save for training
os.makedirs("data", exist_ok=True)
np.savez("data/vx_train.npz", X=X, y=y, data_min=data_min, data_max=data_max)
