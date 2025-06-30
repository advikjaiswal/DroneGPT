import torch
import torch.nn.functional as F
from mini_gpt_model import MiniGPT
import numpy as np

# === Training Hyperparameters ===
batch_size = 4
block_size = 16
max_iters = 100
eval_interval = 10
eval_iters = 2
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load Tokenized Data ===
data = np.memmap('data/train.bin', dtype=np.uint32, mode='r')
vocab_size = int(np.max(data)) + 1
print(f"🧠 Vocab size: {vocab_size}, Block size: {block_size}")

def get_batch(split):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(np.copy(data[i:i+block_size])).long() for i in ix])
    y = torch.stack([torch.from_numpy(np.copy(data[i+1:i+1+block_size])).long() for i in ix])

    return x.to(device), y.to(device)

# === Loss Estimation ===
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split in ['train', 'val']:
        split_losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split="train")  # just use training data for both
            logits = model(xb)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses

# === Initialize Model ===
model = MiniGPT(vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"🔥 Starting training for {max_iters} iterations on device: {device}")

# === Training Loop ===
for iter in range(max_iters):
    xb, yb = get_batch("train")
    logits = model(xb)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"✅ Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    else:
        print(f"🔄 Step {iter}: training... loss {loss.item():.4f}")

# === Save Model ===
torch.save(model.state_dict(), 'model.pt')
print("✅ Model saved to model.pt")


# Save the trained model
torch.save(model.state_dict(), "gpt_action_model.pt")
print("✅ Model saved to gpt_action_model.pt")
