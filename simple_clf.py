import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
from sklearn.model_selection import train_test_split

from network.models.baseline import BaseShip as Network



def train_model(args: argparse.Namespace):
    data_dir = args.data_dir
    hparams = args.hparams
    device = args.device
    model_dir = os.path.join(args.model_dir, 'model.pt')

    X = torch.load(os.path.join(data_dir, 'X.pt'))
    y = torch.load(os.path.join(data_dir, 'y.pt'))

    print("Data shapes (X, y)",X.shape, y.shape)

    with open(hparams, 'r') as f:
        hparams = yaml.safe_load(f)
        network_hparams = hparams['network']
        train_params = hparams['train']

    net = Network(**network_hparams).to(device)

    if args.model_dir:
        print('loading model from', model_dir)
        net.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'))['state_dict'])

    print(f'Loading data from {data_dir}')
    print(f'Loading hyperparameters from {hparams}')
    print(f'Using device {device}')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=float(train_params['lr']))
    # step_size_up = len(train_loader) * 2  # Number of iterations to increase the learning rate
    # step_size_down = len(train_loader) * 2  # Number of iterations to decrease the learning rate
    # scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=step_size_up, step_size_down=step_size_down, mode='triangular')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=train_params['random_seed'])
    train_loader = DataLoader([(x, y) for x, y in zip(X_train, y_train)],shuffle=True)
    test_loader = DataLoader([(x, y) for x, y in zip(X_test, y_test)], shuffle=True)


    n_epochs = train_params['epochs']
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(train_params['epochs']):
        net.train()
        train_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.float().to(device), labels.float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Use the gradient scaler for the forward and backward passes
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            # Scales the loss, and calls backward() on the scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # If the gradients contain infs or NaNs, the optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

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

        print(f'Epoch: {epoch+1}/{n_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--data_dir', type=str, default='/home/rmadeye/kaggle/spaceship/data/inputs')
    parser.add_argument('--hparams', type=str, default='hparams/hparams.yaml')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_model(args)
