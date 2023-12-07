import argparse
import os
import yaml

import torch
import numpy as np
import pandas as pd

from network.models.baseline import BaseShip as Network

def predict(args: argparse.Namespace):
    X_test = torch.load(os.path.join(args.data_dir, 'X_test_kaggle.pt'))
    with open(args.hparams, 'r') as f:
        hparams = yaml.safe_load(f)
        network_hparams = hparams['network']
    net = Network(input_dim=X_test.shape[1], **network_hparams).to(args.device)
    net.load_state_dict(torch.load(args.model_path)['state_dict'])

    print(f"Model loaded from {args.model_path} with parameters: {network_hparams}")
    net.eval()

    results = []
    for i in range(X_test.shape[0]):
        X = X_test[i].unsqueeze(0).float().to(args.device)
        y = net(X)
        transported = (y > 0.5).float()
        results.append(transported.item())

    # loading passenger ids
    passenger_ids = np.load('/home/rmadeye/kaggle/spaceship/data/inputs/passids.npy',
                             allow_pickle=True)
    output = pd.DataFrame({'Transported': results,'PassengerId': passenger_ids})
    output['Transported'] = output['Transported'].astype(bool)
    output.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--hparams', type=str, default='hparams/hparams.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    

    args = parser.parse_args()

    predict(args)