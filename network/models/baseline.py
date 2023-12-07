import torch
from torch import nn
from torch.nn import functional as F

class BaseShip(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=6, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, hidden_state = self.lstm(x)
        # breakpoint()

        # x = self.fc1(x)
        # x = x[:,-1]
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # breakpoint()

        return x.view(-1)
