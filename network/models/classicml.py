import os
import argparse

import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import wandb

def prepare_data(data_dir: str, scaler: bool = True) -> tuple:

    if scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(torch.load(os.path.join(data_dir, 'X.pt')))
        X_val = scaler.transform(torch.load(os.path.join(data_dir, 'X_val.pt')))

    X = torch.load(os.path.join(data_dir, 'X.pt'))
    y = torch.load(os.path.join(data_dir, 'y.pt'))
    X_val = torch.load(os.path.join(data_dir, 'X_val.pt'))
    y_val = torch.load(os.path.join(data_dir, 'y_val.pt'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

    return X_train, X_test, y_train, y_test, X_val, y_val, scaler


def train_logistic_regression(
        data_dir: str,
        penalty: str,
        C: float,
        solver: str,
        max_iter: int,
        multi_class: str,
        class_weight: int,
        use_scaler: bool = True,
        wandb_log: bool = False) -> tuple:
    if wandb_log:
        wandb.init(project="ai-kaggle-titanic", entity="rafal-madaj")

    X_train, X_test, y_train, y_test, X_val, y_val, scaler = prepare_data(data_dir=data_dir, scaler=use_scaler)
    lcf = LogisticRegression(penalty= 'elasticnet' if penalty == 0 else 'l2', 
                             C=C, solver=solver, max_iter=max_iter,
                               multi_class=multi_class, class_weight= None if class_weight == 0 else 'balanced')
    lcf.fit(X_train, y_train)

    train_score = lcf.score(X_train, y_train)
    test_values = lcf.predict(X_test)
    test_score = accuracy_score(y_test, test_values)
    val_values = lcf.predict(X_val)
    val_score = accuracy_score(y_val, val_values)

    results = {'train_score': train_score,
               'test_score': test_score,
               'val_score': val_score,
               'f1_score': f1_score(y_val, val_values, average='weighted')}
    print(results)
    if wandb_log:
        wandb.log(results)

    model = {'model': lcf,
             'scaler': scaler}
    
    return results, model

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str, default='/home/rmadeye/kaggle/spaceship/data/inputs/base')
#     parser.add_argument('--penalty', type=str, default='l2')
#     parser.add_argument('--C', type=float, default=1.0)
#     parser.add_argument('--solver', type=str, default='lbfgs')
#     parser.add_argument('--max_iter', type=int, default=1000)
#     parser.add_argument('--multi_class', type=str, default='auto')
#     parser.add_argument('--class_weight', type=str, default=None)
#     parser.add_argument('--wandb_log', type=bool, default=False)
#     args = parser.parse_args()

#     train_logistic_regression(data_dir=args.data_dir,
#                                 penalty=args.penalty,
#                                 C=args.C,
#                                 solver=args.solver,
#                                 max_iter=args.max_iter,
#                                 multi_class=args.multi_class,
#                                 class_weight=args.class_weight,
#                                 wandb_log=args.wandb_log)