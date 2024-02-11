import os
import argparse
import joblib

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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


def train_knn(
        data_dir: str,
        n_neighbors: int,
        weights: int,
        algorithm: int,
        leaf_size: int,
        p: int,
        use_scaler: bool = True,
        wandb_log: bool = False,
        save_model: bool =  False) -> tuple:
    
    weights = "uniform" if weights == 0 else "distance"
    algorithm = "ball_tree" if algorithm == 0 else "kd_tree" if algorithm == 1 else "brute"
    if wandb_log:
        wandb.init(project="ai-kaggle-titanic", entity="rafal-madaj")



    X_train, X_test, y_train, y_test, X_val, y_val, scaler = prepare_data(data_dir=data_dir, scaler=use_scaler)
    print(X_train.shape)    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, 
                                 weights=weights,
                                 algorithm=algorithm,
                                 leaf_size=leaf_size,
                                 p=p, n_jobs=6).fit(X_train, y_train)




    train_score = knn.score(X_train, y_train)
    test_values = knn.predict(X_test)
    test_score = accuracy_score(y_test, test_values)
    val_values = knn.predict(X_val)
    # val_score = accuracy_score(y_val, val_values)
    accuracy = accuracy_score(y_val, val_values)

    results = {'train_score': train_score,
               'test_score': test_score,
               'accuracy': accuracy,
               'f1_score': f1_score(y_val, val_values, average='weighted')}
    
    print(results)
    if wandb_log:
        wandb.log(results)


    model = {'model': knn,
             'scaler': scaler}
    print(model)
    if save_model:
        joblib.dump(model, os.path.join('/home/rmadeye/kaggle/spaceship/data/outputs', 'rfc.joblib'), compress=0)
    
    return results, model