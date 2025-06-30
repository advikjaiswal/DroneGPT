# train_gpt.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random

# === Hyperparameters ===
block_size = 8     # how many tokens to look back
batch_size = 32
embedding_dim = 32
n_head = 2
n_layer = 2
dropout = 0.1
num_epochs = 1000
eval_interval = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load token data ===
with open("gpt_tokens.pkl", "rb") as f:
    data = pickle.load(f)

encoded = data["encoded"]
vocab_size = data["vocab_size"]

# === Prepare training data ===
def get_batch():
    ix = torch.randint(0, len(encoded) - block_size, (batch_size,))
    x = torch.tensor([encoded[i:i+block_size] for i in ix], dtype=torch.long)
    y = torch.tensor([encoded[i+1:i+block_size+1] for i in ix], dtype=torch.long)
    return x.to(device), y.to(device)

# === Transformer Block ===
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = embedding_dim // n_head
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# === GPT Model ===
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# === Train Model ===
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(num_epochs + 1):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# === Save model ===
torch.save(model.state_dict(), "gpt_action_model.pt")
print("✅ GPT model saved.")
# train_gpt.py
import torch
import numpy as np
import pickle
from mini_gpt_model import MiniGPT

# === Load Data ===
train_data = np.fromfile("data/train.bin", dtype=np.uint16)
val_data = np.fromfile("data/val.bin", dtype=np.uint16)

with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)

itos = meta["itos"]
stoi = meta["stoi"]
vocab_size = meta["vocab_size"]

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")
print(f"Vocab size: {vocab_size}")

# === Hyperparameters ===
block_size = 8
batch_size = 4


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# === Initialize the model ===
model = MiniGPT(vocab_size, block_size)

# === Forward Pass Test ===
xb, yb = get_batch("train")  # input and target
logits = model.forward(xb)
print(f"logits shape: {logits.shape}")  # should be (B, T, vocab_size)
