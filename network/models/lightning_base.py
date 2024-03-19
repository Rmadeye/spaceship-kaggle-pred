import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn


class LightningBase(pl.LightningModule):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3, *args, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//4)
        self.fc4 = nn.Linear(hidden_dim//4, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)   
        self.bn2 = nn.BatchNorm1d(hidden_dim//4)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.l1_lambda  = 0


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn2(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.fc4(x))
        x = torch.sigmoid(x)
        return x.view(-1)


    def binarize(self, x):
        return torch.Tensor([0 if z < 0.5 else 1 for z in x]).to(self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.binarize(self(x))
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat == y).sum().item() / y.size(0)
        self.log('test_loss', loss,  logger=True, on_epoch=True)
        self.log('test_acc', accuracy, logger=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.binarize(self(x))
        loss = self.criterion(y_hat, y) 
        accuracy = (y_hat == y).sum().item() / y.size(0)
        self.log('val_loss', loss, prog_bar=True, logger=False, on_epoch=True)
        self.log('val_acc', accuracy, prog_bar=True, logger=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def criterion(self, y_hat, y):
        return nn.CrossEntropyLoss()(y_hat, y)s