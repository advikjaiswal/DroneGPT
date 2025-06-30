import numpy as np
import torch

from train_predictor import VelocityMLP

# Load data for normalization
data = np.load("data/vx_train.npz")
X = data["X"]
data_min = data["data_min"]
data_max = data["data_max"]

# Pick last sequence as input
input_seq = torch.tensor(X[-1:], dtype=torch.float32)

# Load model
model = VelocityMLP()
model.load_state_dict(torch.load("vx_predictor.pt"))
model.eval()

# Predict normalized output
with torch.no_grad():
    pred = model(input_seq).item()

# Convert back to original scale
vx_predicted = pred * (data_max - data_min) + data_min
print(f"Predicted vx (m/s): {vx_predicted:.3f}")
