#!/bin/bash

hparams=hparams/hparams_base_2.yaml
device=cuda
data=/home/rmadeye/kaggle/spaceship/data/inputs/dropna_with_val/
output_dir=/home/rmadeye/kaggle/spaceship/data/outputs
model_dir=''
EXEC=simple_clf.py
python3 "$EXEC" --hparams $hparams --device $device --data $data --model_dir=$model_dir --output_dir=$output_dir
