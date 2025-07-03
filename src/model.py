import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=128, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden_dim*2, num_classes)
    def forward(self, x):
        # x: (batch, frames, features)
        out, _ = self.lstm(x)
        # take last time step
        out = out[:, -1, :]           # (batch, hidden*2)
        return self.fc(out)           # (batch, num_classes)
