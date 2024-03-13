![Python Version](https://img.shields.io/badge/Python-3.12-blue) ![NumPy Version](https://img.shields.io/badge/NumPy-latest-green)![scikit-learn Version](https://img.shields.io/badge/scikit--learn-latest-yellowgreen)![PyTorch Version](https://img.shields.io/badge/PyTorch-newest-red)
## Spaceship Titanic survival prediction

Repository contains simple model for predicting survival of passengers on Titanic Spaceship competition on kaggle [https://www.kaggle.com/competitions/spaceship-titanic]

For now there are two approaches being used:
- Deep learning model based on classical linear layers
- Machine learning models which include:
    - Random Forest Classifier
    - LGBM
    - KNN

Also several approaches to data preprocessing are being used:

- Dropping all nan values
- Filling nan values with mean/median
- Statistical approach combined with KNN imputations.

## Current results

| Model  | Best score |
|-------|--------------|
| Linear model (DL)    | 79.4 % | 
| Random Forest Classifier | 79.6 % |
| LGBM | 78.6 % |
| KNN | 76.6 % |

## DL architecture

Model is based on simple linear layers with dropout and batch normalization. It is trained using Adam optimizer and binary cross entropy loss function.
Currently it consists of 3 layers with 128, 32 and 1 neuron, respectively.

## To do
[ ] Add predictor for ML  algorithms
[ ] Add more preprocessing methods
[ ] Add more models

