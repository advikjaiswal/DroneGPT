import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPredictor(nn.Module):
    def __init__(self, context_window):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_window, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predict single value
        )

    def forward(self, x):
        return self.net(x)
