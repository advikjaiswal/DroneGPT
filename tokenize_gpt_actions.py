import os
import csv
import pickle

DATA_DIR = "drone_data"
OUTPUT_TOKENS_FILE = "vx_action_tokens.txt"
OUTPUT_META_FILE = "vx_action_meta.pkl"

# Simple rule-based vx to action converter
def vx_to_action(vx):
    try:
        vx = float(vx)
    except:
        return None
    if abs(vx) < 0.5:
        return "Hover"
    elif vx > 0:
        return "Forward"
    else:
        return "Backward"

# Collect tokens
tokens = []
vocab = set()

print("🔍 Scanning drone CSV files...")

for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".csv"):
        continue
    path = os.path.join(DATA_DIR, filename)

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if "velocity[0]" not in reader.fieldnames:
            print(f"⚠️ Skipping {filename} (missing vx column)")
            continue

        vx_col = "velocity[0]"
        file_tokens = []

        for row in reader:
            vx = row.get(vx_col, "")
            action = vx_to_action(vx)
            if action is None:
                continue
            token = f"{round(float(vx), 2)}|{action}"
            file_tokens.append(token)
            vocab.add(token)

        if file_tokens:
            tokens.extend(file_tokens)
            print(f"✅ Processed {filename} ({len(file_tokens)} tokens)")
        else:
            print(f"⚠️ Skipped {filename} (no valid vx rows)")

# Build vocab and mappings
vocab = sorted(list(vocab))
stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for s, i in stoi.items()}
encoded = [stoi[t] for t in tokens]

# Save tokens
with open(OUTPUT_TOKENS_FILE, "w") as f:
    f.write(" ".join(map(str, encoded)))

# Save vocab
with open(OUTPUT_META_FILE, "wb") as f:
    pickle.dump((stoi, itos), f)

print(f"\n✅ Tokenization complete.")
print(f"📦 Total tokens: {len(tokens)}")
print(f"🔠 Vocab size: {len(vocab)}")
print(f"🔢 Sample tokens: {tokens[:10]}")
