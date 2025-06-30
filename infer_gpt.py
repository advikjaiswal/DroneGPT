# infer_gpt.py

import torch
from model import GPTConfig, GPT
import pickle

# Load metadata
with open("vx_action_meta.pkl", "rb") as f:
    meta = pickle.load(f)
    stoi = meta["stoi"]
    itos = meta["itos"]

vocab_size = len(stoi)
device = "cpu"

# ✅ Build the model using GPTConfig
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=8,
    n_layer=2,
    n_head=2,
    n_embd=32,
)
model = GPT(config).to(device)

# Load trained weights
model.load_state_dict(torch.load("gpt_action_model.pt", map_location=device), strict=False)
model.eval()

# Tokenize a sample context
context = ['0.1|Hover', '0.1|Hover', '0.2|Hover', '0.2|Hover']
context_ids = [stoi.get(token, 0) for token in context]
print("🧠 Initial context token IDs:", context_ids)

# Generate new tokens
input_tensor = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
with torch.no_grad():
    output_ids = model.generate(input_tensor, max_new_tokens=20)[0].tolist()

print("🧠 Final generated token IDs:", output_ids)

# Decode to readable text
predicted_tokens = [itos[i] for i in output_ids]
predicted_actions = [tok.split('|')[-1] for tok in predicted_tokens]

print("🧠 GPT Predicted Drone Actions:")
print(predicted_actions)
