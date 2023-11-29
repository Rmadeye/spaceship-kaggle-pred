#!/bin/bash

hparams=hparams/hparams_base.yaml
device=cuda
data=data/inputs/data_ext/
model_dir=''
EXEC=simple_clf.py
python3 "$EXEC" --hparams $hparams --device $device --data $data --model_dir=$model_dir