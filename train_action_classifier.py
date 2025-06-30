# train_action_classifier.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the labeled data
data = np.load("labeled_actions.npz")
X, y = data["X"], data["y"]

# Encode string labels ("Hover", "None", etc.) to integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the neural network
class ActionClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, len(encoder.classes_))  # Output = number of unique labels
        )

    def forward(self, x):
        return self.net(x)

model = ActionClassifier(X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Evaluate on test set
with torch.no_grad():
    test_outputs = model(X_test)
    predicted = torch.argmax(test_outputs, dim=1)
    acc = (predicted == y_test).float().mean().item()
    print("Test accuracy:", round(acc * 100, 2), "%")

# Save model
torch.save(model.state_dict(), "action_classifier.pth")
print("Model saved. Label classes:", encoder.classes_)
