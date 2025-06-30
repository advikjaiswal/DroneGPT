import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load labeled data
data = np.load("labeled_actions.npz")
X = data['X']
y = data['y']

# 2. Load original timestamps from the CSV
df = pd.read_csv("drone_data/3ddb3350-469e-4508-a887-59b5531f4487_vehicle_local_position_0.csv")
timestamps = df['timestamp'].values

# Adjust timestamps to match labeled sequences
seq_len = X.shape[1]
adjusted_timestamps = timestamps[seq_len:]

# 3. Convert labels to integers for plotting
label_to_int = {label: idx for idx, label in enumerate(np.unique(y))}
int_labels = [label_to_int[label] for label in y]

# Plot the actions over time
plt.figure(figsize=(12, 4))
plt.plot(adjusted_timestamps, int_labels, label="Action Label", marker='o', linestyle='--')
plt.yticks(ticks=list(label_to_int.values()), labels=list(label_to_int.keys()))
plt.title("Predicted Drone Actions Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Predicted Action")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
