import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import wandb

from network.models.baseline import BaseShip as Network

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(data_dir: str,
                 batch_size: int,
                 ):
    print(data_dir)
    X = torch.load(os.path.join(data_dir, 'X.pt'))
    y = torch.load(os.path.join(data_dir, 'y.pt'))
    X_val = torch.load(os.path.join(data_dir, 'X_val.pt'))
    y_val = torch.load(os.path.join(data_dir, 'y_val.pt'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(MyDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader





def train_model(args: argparse.Namespace):
    data_dir = args.data_dir
    hparams = args.hparams
    device = args.device
    # if args.wandb_log:
    #     wandb.init(project="ai-kaggle-titanic", entity="rafal-madaj")

    X = torch.load(os.path.join(data_dir, 'X.pt'))
    y = torch.load(os.path.join(data_dir, 'y.pt'))



    print("Data shapes (X, y)",X.shape, y.shape)

    with open(hparams, 'r') as f:
        hparams = yaml.safe_load(f)
        network_hparams = hparams['network']
        train_params = hparams['train']

    net = Network(input_dim=X.shape[1],**network_hparams).to(device)

    if args.model_path:
        print('loading model from', args.model_path)
        net.load_state_dict(torch.load(args.model_path)['state_dict'])

    print(f'Loading data from {data_dir}')
    print(f'Loading hyperparameters from {hparams}')
    print(f'Using device {device}')
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=float(train_params['lr']), weight_decay=float(train_params['weight_decay']))
    scheduler= ReduceLROnPlateau(optimizer, 'min', patience=train_params['lr_patience'], verbose=True)


    train_loader, test_loader, val_loader = prepare_data(data_dir, train_params['batch_size'])
    n_epochs = train_params['epochs']
    loss = float('inf')
    best_test_error = 0.65
    num_epochs_without_gain = 0
    for epoch in range(n_epochs):
        net.train()

        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            l1_loss = net.l1_loss()
            total_loss = loss.item() + l1_loss
            train_loss += total_loss  #loss.item()
            loss.backward()
            optimizer.step()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 3)

        train_loss = train_loss / len(train_loader.dataset)
        scheduler.step(train_loss)

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.float().to(device), labels.to(device)

                outputs = net(inputs)
                predicted = (outputs > 0.5).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

            # Checkpointing and early stopping based on test error

        print(f"Current acc vs best acc: {round(accuracy,2)} vs {round(best_test_error,2)}")
        if best_test_error < accuracy:
            best_test_error = accuracy
            checkpoint_dir = os.path.join(args.output_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {'state_dict' :net.state_dict(),
                'hparams' : hparams,
                'trainstats': {'epoch': epoch,
                                'accuracy': accuracy,
                                'loss': loss }},
                os.path.join(checkpoint_dir, f'model_2.pt')
                )                                       
            num_epochs_without_gain = 0
        else:
            num_epochs_without_gain += 1
        if num_epochs_without_gain >= train_params['early_stopping']:
            print(f'early stopping after {epoch}')
            break

        print(f'Epoch: {epoch+1}/{n_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
    # now validation on val set
    net.eval()
    correct = 0
    total = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.float().to(device), labels.to(device)

        outputs = net(inputs)
        predicted = (outputs > 0.5).float()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on validation set: {correct / total * 100:.2f}%')
    accuracy = round(accuracy*100, 2)
    val_acc = round(correct / total * 100, 2)
    results = {'best_epoch_acc': accuracy,
            'validation_acc': val_acc}
    # if args.wandb_log:
    #     wandb.log(results)
    with open("results_log.txt", "a") as f:
        f.write(str(results) + "\n")
        f.write("hparams: " + str(hparams) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--data_dir', type=str, default='/home/rmadeye/kaggle/spaceship/data/inputs')
    parser.add_argument('--hparams', type=str, default='hparams/hparams.yaml')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='saved_models')
    parser.add_argument('--wandb_log', type=bool, default=False)
    args = parser.parse_args()
    train_model(args)
