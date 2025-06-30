import pickle
import matplotlib.pyplot as plt
from model import GPT, GPTConfig
import torch

# === Load vocab mappings ===
with open("vx_action_meta.pkl", "rb") as f:
    stoi, itos = pickle.load(f)

vocab_size = len(itos)

# === Model configuration ===
block_size = 128
n_layer = 4
n_head = 4
n_embd = 32
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd
)

# === Load model ===
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT(config)
model.load_state_dict(torch.load("gpt_model.pt", map_location=device))
model.to(device)
model.eval()

# === Load token data ===
with open("vx_action_tokens.bin", "rb") as f:
    data = torch.frombuffer(f.read(), dtype=torch.int32)

print(f"✅ Vocab size: {vocab_size}")
print(f"📦 Total tokens: {len(data)}")

# === Prepare context input ===
context = data[:8].unsqueeze(0)  # (1, block_size)

# === Generate predictions ===
with torch.no_grad():
    out = model.generate(context.to(device), max_new_tokens=200)[0].tolist()

# === Decode 'vx|action' tokens ===
decoded = [itos[i] for i in out]

vx_values = []
actions = []

for token in decoded:
    try:
        vx_str, action = token.split("|")
        vx = float(vx_str)
        vx_values.append(vx)
        actions.append(action)
    except Exception:
        continue

# === Plot ===
plt.figure(figsize=(12, 5))
plt.plot(vx_values, label="vx", linewidth=2, color="black")

action_colors = {
    "Hover": "orange",
    "Move": "green",
    "Return": "red",
    "None": "gray",
}

for i, action in enumerate(actions):
    color = action_colors.get(action, "gray")
    plt.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.2)

plt.xlabel("Timestep")
plt.ylabel("vx (forward velocity)")
plt.title("GPT Predicted Drone Actions by vx")
handles = [plt.Line2D([0], [0], color=color, lw=6, alpha=0.4, label=label)
           for label, color in action_colors.items()]
plt.legend(handles=handles)
plt.grid(True)
plt.tight_layout()
plt.savefig("gpt_vx_actions.png")
plt.show()
