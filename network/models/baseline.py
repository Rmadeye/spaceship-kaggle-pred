import torch
from torch import nn
from torch.nn import functional as F

class BaseShip(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3, *args, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//4)
        self.fc3 = nn.Linear(hidden_dim//4, 1)  
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(x)
        return x.view(-1)
