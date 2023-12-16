#!/bin/bash

hparams=hparams/hparams.yaml
device=cuda
data=data/inputs/new_base
output_dir=/home/rmadeye/kaggle/spaceship/data/outputs
# model_dir='/home/rmadeye/kaggle/spaceship/data/outputs/model.pt'
EXEC=simple_clf.py
python3 "$EXEC" --hparams $hparams --device $device --data $data --model_path=$model_dir --output_dir=$output_dir
