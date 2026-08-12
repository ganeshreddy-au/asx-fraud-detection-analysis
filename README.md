# ASX Fraud Detection Analysis

This project applies machine learning to two public datasets to explore anomaly and fraud detection, connected to a Data Scientist role at ASX focused on participant supervision and market integrity.

## What this project does

Two machine learning models, Random Forest and Naive Bayes, are trained and compared on two different datasets:

1. Credit Card Fraud Detection, a labelled dataset of real transactions where a small percentage are fraud.
2. AAPL Stock Price and Volume, daily trading data used to predict whether the next day's closing price goes up or down.

The goal is to see how well each model performs on a problem with a clear signal (credit card fraud) compared to a much harder problem where the signal may not exist at all (predicting daily stock direction).

## Datasets used

- Credit Card Fraud Detection (Kaggle, ULB): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Huge Stock Market Dataset (Kaggle): https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
  - Only the AAPL file (`aapl.us.txt`) from this dataset was used for this project, not the full collection of stocks.

Datasets are not included in this repository due to file size. Download them directly from the links above if you want to run the notebooks yourself. For the stock dataset, only `aapl.us.txt` is needed.

## How to run
1. Download the datasets from the links above
2. Place `creditcard.csv` and `aapl.us.txt` in the same folder as the notebooks
3. Open `credit_card_fraud_analysis.ipynb` in Jupyter and run all cells

## Files in this repository

- `credit_card_fraud_analysis.ipynb`, code and results for the Credit Card Fraud dataset
- `aapl_stock_analysis.ipynb`, code and results for the AAPL stock dataset
- `roc_dataset1_creditcard.png`, ROC curve comparing both models on the Credit Card dataset
- `roc_dataset2_aapl.png`, ROC curve comparing both models on the AAPL dataset

## Methods used

- Data cleaning and feature scaling with StandardScaler
- SMOTE to handle class imbalance on the Credit Card dataset
- Random Forest and Naive Bayes classifiers
- Evaluation using precision, recall, F1 score and ROC AUC, chosen over accuracy due to class imbalance and the cost of missing real fraud cases

## Results summary

The Credit Card dataset produced strong results for both models, with Random Forest reaching a ROC AUC of 0.973. The AAPL dataset produced results close to random guessing for both models, showing that daily stock direction is very hard to predict using price and volume alone.

## Author

Ganesh Kumar Reddy Avula
Master of Data Science, RMIT University