import pickle
import numpy as np
import os

# Load encoded data
with open("gpt_tokens.pkl", "rb") as f:
    data = pickle.load(f)
    encoded = data["encoded"]

# Train/val split
n = int(0.9 * len(encoded))
train_ids = np.array(encoded[:n], dtype=np.uint32)
val_ids = np.array(encoded[n:], dtype=np.uint32)

# Make output dir
os.makedirs("data", exist_ok=True)

# Save to binary files
train_ids.tofile("data/train.bin")
val_ids.tofile("data/val.bin")

print("✅ Saved train.bin and val.bin")
