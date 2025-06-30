import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ✅ Classifier model defined directly here
class ClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ClassifierModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load the labeled vx sequence from your real drone log
df = np.load("labeled_actions.npz")
X = df["X"]
y = df["y"]

# Normalize and convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)

# 🔁 Load your trained model
model = ClassifierModel(input_size=16, hidden_size=32, num_classes=2)
model.load_state_dict(torch.load("vx_classifier.pt"))
model.eval()

# ✅ Predict class indices over time
with torch.no_grad():
    outputs = model(X_tensor)
    predictions = torch.argmax(outputs, dim=1).numpy()

# Map index to label
index_to_label = {0: "Hover", 1: "None"}
labels = np.array([index_to_label[p] for p in predictions])

# ✅ Plot classification results over time
plt.figure(figsize=(12, 4))
plt.plot(labels, marker='o', linestyle='', label='Predicted Action')
plt.title("Predicted Actions Over Time")
plt.xlabel("Time Step")
plt.ylabel("Action")
plt.yticks(ticks=[0, 1], labels=["Hover", "None"])
plt.grid(True)
plt.tight_layout()
plt.show()
