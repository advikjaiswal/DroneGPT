# rebuild_tokens.py

import pickle

with open("gpt_action_training.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

with open("gpt_tokens.pkl", "wb") as f:
    pickle.dump({
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size
    }, f)

print(f"✅ gpt_tokens.pkl rebuilt with {vocab_size} tokens.")
