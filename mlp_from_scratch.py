import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Load labeled data
data = np.load("labeled_actions.npz", allow_pickle=True)
X = torch.tensor(data["X"], dtype=torch.float32)
y_raw = data["y"]

# Encode string labels to integers
label_names = np.unique(y_raw)
label_map = {label: idx for idx, label in enumerate(label_names)}
y = torch.tensor([label_map[label] for label in y_raw], dtype=torch.long)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Match this architecture in both train + inference
class ClassifierModel(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

model = ClassifierModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    test_outputs = model(X_test)
    predicted = torch.argmax(test_outputs, dim=1)
    acc = (predicted == y_test).float().mean().item()
    print(f"Final Accuracy: {acc * 100:.2f}%")

# Save model + label map
torch.save(model.state_dict(), "vx_classifier.pt")
np.save("label_names.npy", label_names)
print("✅ Model saved to vx_classifier.pt")
print("✅ Label classes saved:", label_names)
