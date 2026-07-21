# Purchase Prediction

ML project predicting whether an online shopper will make a purchase 
based on their session behavior, using Logistic Regression and Random Forest.

## Goal
Build a binary classification model to predict purchase intent (Revenue: True/False) 
from user session data — page visits, duration, bounce rates, and other behavioral signals.

## Dataset
Source: Online Shoppers Purchasing Intention Dataset (https://www.kaggle.com/datasets/henrysue/online-shoppers-intention) 

Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction 
of online shoppers' purchasing intention using multilayer perceptron 
and LSTM neural networks. Neural Comput & Applic (2018).

12,330 sessions, 18 features, binary target (Revenue: True/False).

## Approach
- Exploratory Data Analysis (EDA) — class balance, distributions, correlations
- Data cleaning and preprocessing — encoding categorical features
- Train/test split
- Model 1: Logistic Regression
- Model 2: Random Forest
- Evaluation: accuracy, precision, recall, confusion matrix
- Feature importance analysis

## Results


## How to run
Open `purchase_prediction.ipynb` in Google Colab. Dataset is included in this repo.
