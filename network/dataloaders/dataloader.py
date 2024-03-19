import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


from sklearn.model_selection import train_test_split
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataModel(pl.LightningDataModule):
    def __init__(self, data_dir='/home/rmadeye/kaggle/spaceship/data/inputs', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        X = torch.load(os.path.join(self.data_dir, 'X.pt'))
        y = torch.load(os.path.join(self.data_dir, 'y.pt'))
        X_val = torch.load(os.path.join(self.data_dir, 'X_val.pt'))
        y_val = torch.load(os.path.join(self.data_dir, 'y_val.pt'))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

        train_dataset = MyDataset(X_train, y_train)
        test_dataset = MyDataset(X_test, y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(MyDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)  

        self.dims = self.test_loader.dataset.X.shape, self.val_loader.dataset.X.shape

    def train_dataloader(self):
        print(f"Train shape: {self.train_loader.dataset.X.shape}, test shape: {self.test_loader.dataset.X.shape}, val shape: {self.val_loader.dataset.X.shape}")
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

    def test_dataloader(self):
        return self.val_loader