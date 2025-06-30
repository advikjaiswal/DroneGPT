import numpy as np
import pandas as pd

def label_action(vx_seq):
    recent = vx_seq[-1]
    prev = vx_seq[-2] if len(vx_seq) > 1 else 0
    diff = abs(recent - prev)
    
    if abs(recent) > 5 and diff > 3:
        return "RTH"
    elif abs(recent) < 0.5 and np.all(np.abs(vx_seq) < 0.5):
        return "Hover"
    elif diff > 8:
        return "Kill"
    else:
        return "None"

def create_labeled_sequences(vx_array, seq_length=16):
    X = []
    y = []
    for i in range(len(vx_array) - seq_length):
        seq = vx_array[i:i + seq_length]
        label = label_action(seq)
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

# Example:
df = pd.read_csv("drone_data/3ddb3350-469e-4508-a887-59b5531f4487_vehicle_local_position_0.csv")  # Replace with your CSV
vx = df['vx'].dropna().to_numpy()
X, y = create_labeled_sequences(vx)

# Save for training
np.savez("labeled_actions.npz", X=X, y=y)
print("Done labeling:", np.unique(y, return_counts=True))
