# Purchase Prediction

ML project predicting whether an online shopper will make a purchase 
based on their session behavior, using Logistic Regression and Random Forest.

## Goal
Build a binary classification model to predict purchase intent (Revenue: True/False) 
from user session data — page visits, duration, bounce rates, and other behavioral signals.

## Dataset
Source: [Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention)

Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction 
of online shoppers' purchasing intention using multilayer perceptron 
and LSTM neural networks. Neural Comput & Applic (2018).

12,330 sessions, 18 features, binary target (Revenue: True/False). 
After removing 125 duplicates: 12,205 rows.

## Approach
- Exploratory Data Analysis (EDA) — class balance, distributions, correlations
- Data preprocessing — encoding categorical features (Month, VisitorType), 
  converting boolean columns to 0/1, removing duplicates
- Train/test split (80/20)
- Model 1: Logistic Regression (class_weight='balanced')
- Model 2: Random Forest (class_weight='balanced')
- Evaluation: accuracy, precision, recall, confusion matrix
- Feature importance analysis

## Results

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 83% | 46% | 81% |
| Random Forest | 90% | 75% | 53% |

Logistic Regression catches more actual buyers (higher recall), at the cost 
of more false positives. Random Forest is more accurate overall and more 
confident in its "will buy" predictions, but misses more buyers.

## Feature Importance

PageValues is by far the strongest predictor (37.6%), more than 3x the next 
feature. Product-browsing behavior matters more than demographic/technical features.

## How to run
Open `purchase_prediction.ipynb` in Google Colab. Dataset is included in this repo.
