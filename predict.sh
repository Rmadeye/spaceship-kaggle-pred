#!/bin/bash

data_dir=data/inputs/knnfill
model_path=data/outputs/model_2.pt
hparams=hparams/hparams.yaml

python3 predict.py --data_dir $data_dir --model_path $model_path

kaggle competitions submit -c spaceship-titanic -f submission.csv -m "$(cat $hparams)"