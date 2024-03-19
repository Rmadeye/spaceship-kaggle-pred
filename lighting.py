import os
import argparse

import yaml
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from network.dataloaders.dataloader import DataModel  
from network.models.lightning_base import LightningBase as LB
from pytorch_lightning.callbacks import LearningRateFinder

class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


def train_lightning_model(data_dir='data/inputs/knnfill',
                          hparams='hparams/hparams.yaml',
                          model_path='models',
                          device='cuda',
                          output_dir='saved_models',
                          wandb_log=False):

    data_module = DataModel(data_dir=data_dir)
    with open(hparams, 'r') as f:
        hparams = yaml.safe_load(f)
        network_hparams = hparams['network']
        train_params = hparams['train']
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_acc", min_delta=0.00, 
        patience=train_params['lr_patience'], verbose=True, mode="max")
    model = LB(12 , **network_hparams)

    trainer = pl.Trainer(max_epochs=train_params['epochs'], callbacks=[early_stop_callback, FineTuneLearningRateFinder(milestones=(5,10))],
    default_root_dir ='lighting_weights')
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, dataloaders=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--data_dir', type=str, default='data/inputs/knnfill')
    parser.add_argument('--hparams', type=str, default='hparams/hparams.yaml')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='saved_models')
    parser.add_argument('--wandb_log', type=bool, default=False)
    args = parser.parse_args()
    train_lightning_model(data_dir=args.data_dir,
                          hparams=args.hparams,
                          model_path=args.model_path,
                          device=args.device,
                          output_dir=args.output_dir,
                          wandb_log=args.wandb_log)


