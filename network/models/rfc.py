import os
import argparse

import torch
from sklearn.model_selection import train_test_split
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


def train_rfc(
        data_dir: str,
        n_estimators: int,
        max_depth: int,
        criterion: str,
        max_features: int,
        class_weight: int,
        ccp_alpha: float,
        bootstrap: bool,
        use_scaler: bool = True,
        wandb_log: bool = False) -> tuple:
    
    max_features = None if max_features == 0 else 'sqrt' if max_features == 1 else 'log2'
    class_weight = None if class_weight == 0 else 'balanced' if class_weight == 1 else 'balanced_subsample'
    if wandb_log:
        wandb.init(project="ai-kaggle-titanic", entity="rafal-madaj")



    X_train, X_test, y_train, y_test, X_val, y_val, scaler = prepare_data(data_dir=data_dir, scaler=use_scaler)
    rfc = RandomForestClassifier(n_estimators=n_estimators, 
                                 max_depth=max_depth,
                                 criterion=criterion,
                                 class_weight=class_weight,
                                 ccp_alpha=ccp_alpha,
                                 bootstrap=bootstrap, 
                                 random_state=666).fit(X_train, y_train)




    train_score = rfc.score(X_train, y_train)
    test_values = rfc.predict(X_test)
    test_score = accuracy_score(y_test, test_values)
    val_values = rfc.predict(X_val)
    # val_score = accuracy_score(y_val, val_values)
    accuracy = accuracy_score(y_val, val_values)

    results = {'train_score': train_score,
               'test_score': test_score,
               'accuracy': accuracy,
               'f1_score': f1_score(y_val, val_values, average='weighted')}
    
    print(results)
    if wandb_log:
        wandb.log(results)

    model = {'model': rfc,
             'scaler': scaler}
    
    return results, model


