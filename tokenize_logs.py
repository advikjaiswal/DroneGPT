# tokenize_logs.py
import os
import csv
import numpy as np
from glob import glob

log_dir = "drone_data"
merged_tokens = []

# Load and tokenize each CSV
for file in glob(f"{log_dir}/*.csv"):
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = []
            for key, value in row.items():
                try:
                    val = float(value)
                    val = round(val, 2)
                    token = f"{key}={val}"
                except ValueError:
                    token = f"{key}={value}"
                tokens.append(token)
            merged_tokens.extend(tokens)

# === Create Vocabulary ===
unique_tokens = sorted(set(merged_tokens))
stoi = {tok: i for i, tok in enumerate(unique_tokens)}
itos = {i: tok for tok, i in stoi.items()}
vocab_size = len(stoi)
print(f"🧠 Final vocab size: {vocab_size}")

# === Tokenize Full Dataset ===
ids = [stoi[tok] for tok in merged_tokens]
ids = np.array(ids, dtype=np.uint32)  # ← 🧠 use uint32 to support larger vocab

# === Train/Val Split ===
split = int(0.9 * len(ids))
train_ids = ids[:split]
val_ids = ids[split:]

train_ids.tofile("data/train.bin")
val_ids.tofile("data/val.bin")

# Save vocab for decoding
with open("data/drone_tokens.txt", "w") as f:
    for tok in unique_tokens:
        f.write(tok + "\n")

print(f"✅ Tokenized: {len(train_ids)} train tokens, {len(val_ids)} val tokens")
