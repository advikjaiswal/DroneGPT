import pickle
import numpy as np
import os

with open("gpt_tokens.pkl", "rb") as f:
    data = pickle.load(f)
    encoded = np.array(data["encoded"], dtype=np.uint32)
    stoi = data["token_to_id"]
    itos = data["id_to_token"]

# Train/val split
n = int(0.9 * len(encoded))
train_data = encoded[:n]
val_data = encoded[n:]

os.makedirs("data", exist_ok=True)

# Save .bin files
train_data.tofile("data/train.bin")
val_data.tofile("data/val.bin")

# Save meta info
meta = {
    "vocab_size": len(stoi),
    "itos": itos,
    "stoi": stoi,
}
with open("data/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("✅ Saved train.bin, val.bin, meta.pkl")
