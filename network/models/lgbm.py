import os
import argparse
import joblib

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import wandb

def prepare_data(data_dir: str, scaler: bool = True, final_train: bool = False) -> tuple:

    if scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(torch.load(os.path.join(data_dir, 'X.pt')))
        X_val = scaler.transform(torch.load(os.path.join(data_dir, 'X_val.pt')))

    X = torch.load(os.path.join(data_dir, 'X.pt'))
    y = torch.load(os.path.join(data_dir, 'y.pt'))
    X_val = torch.load(os.path.join(data_dir, 'X_val.pt'))
    y_val = torch.load(os.path.join(data_dir, 'y_val.pt'))

    if final_train:
        return torch.concat((X, X_val)), None, torch.concat((y, y_val)), None, None, None, scaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

    return X_train, X_test, y_train, y_test, X_val, y_val, scaler


def train_lgbm(
        data_dir: str,
        boosting_type: int = 0,
        num_leaves: int = 30,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: str = 'binary',
        # class_weight: dict = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 0.001,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int = None,
        n_jobs: int = 6,
        importance_type: int = 0,
        use_scaler: bool = True,
        wandb_log: bool = False,
        save_model: bool = False,
        final_train: bool = False) -> tuple:
    
    boosting_type = "gbdt" if boosting_type == 0 else "dart" if boosting_type == 1 else "goss" if boosting_type == 2 else "rf"
    # importance_type = "split" if importance_type == 0 else "gain"
    # class_weight = None if class_weight == 0 else 'balanced' if class_weight == 1 else 'balanced_subsample'

    if wandb_log:
        wandb.init(project="ai-kaggle-titanic", entity="rafal-madaj")


    
    X_train, X_test, y_train, y_test, X_val, y_val, scaler = prepare_data(data_dir=data_dir, scaler=use_scaler) 
    lgbm = LGBMClassifier(
        # boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        subsample_for_bin=subsample_for_bin,
                        objective=objective,
                        # class_weight=class_weight,
                        min_split_gain=min_split_gain,
                        min_child_weight=min_child_weight,
                        min_child_samples=min_child_samples,
                        subsample=subsample,
                        subsample_freq=subsample_freq,
                        colsample_bytree=colsample_bytree,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        importance_type=importance_type
                        ).fit(X_train, y_train)

    if final_train:
        model = {'model': lgbm,
            'scaler': scaler}
        joblib.dump(model, os.path.join('/home/rmadeye/kaggle/spaceship/data/outputs', 'lgb.joblib'), compress=0)
        return "Final model saved"
        
    



    train_score = lgbm.score(X_train, y_train)
    test_values = lgbm.predict(X_test)
    test_score = accuracy_score(y_test, test_values)
    val_values = lgbm.predict(X_val)
    # val_score = accuracy_score(y_val, val_values)
    accuracy = accuracy_score(y_val, val_values)

    results = {'train_score': train_score,
               'test_score': test_score,
               'accuracy': accuracy,
               'f1_score': f1_score(y_val, val_values, average='weighted')}
    
    print(results)
    if wandb_log:
        wandb.log(results)


    model = {'model': lgbm,
             'scaler': scaler}
    print(model)
    if save_model:
        joblib.dump(model, os.path.join('/home/rmadeye/kaggle/spaceship/data/outputs', 'lgb.joblib'), compress=0)
    
    return results, model