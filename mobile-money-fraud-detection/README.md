# Mobile-Money Fraud Detection (DS-09)
**3MTT NextGen Fellowship — Data Science Track — Capstone Project (MVP)**

## 1. Problem Statement

Mobile-money platforms (OPay, Paga, MTN MoMo, PalmPay, etc.) process millions
of naira in daily transfers, cash-outs, and payments. Fraud — account
takeover, mule-account laundering, and rapid SIM-swap-style draining —
costs users and providers significant money and erodes trust in the
platform. This project builds a **machine learning model that scores
each transaction with a fraud probability in real time**, so suspicious
transactions can be flagged for review or blocked before completion.

## 2. Data

Real mobile-money transaction logs are private and were not available for
this project, so a **synthetic dataset of ~61,000 transactions** was
generated (`src/generate_data.py`) to closely mirror the structure and
statistical shape of real mobile-money data (transaction type, amount,
sender/recipient balances before & after, timestamp). Three realistic
fraud typologies were injected at a ~1.5% rate (close to real-world fraud
prevalence):

| Pattern | Description |
|---|---|
| **Account takeover** | A single large transfer/cash-out that drains an account almost completely, often at an odd hour. |
| **Mule collusion** | A transfer to a new/unverified "mule" account, immediately cashed out to launder the funds. |
| **Rapid micro-fraud burst** | Several small transfers in quick succession, typical of SIM-swap style abuse. |

> **Note:** Because this is synthetic data, model performance here should
> be read as a proof-of-concept. A production version would need to be
> re-trained and validated on real, labeled transaction data from the
> platform before deployment.

## 3. Approach

1. **Feature engineering** (`src/train_model.py → engineer_features`):
   - Balance-consistency errors (`errorBalanceOrig`, `errorBalanceDest`) — legitimate transactions should satisfy `old_balance - amount ≈ new_balance`; fraud often breaks this.
   - `drainRatioOrig` — how much of the sender's balance the transaction consumes.
   - `accountEmptied` — flag when the sender's balance goes to ~0.
   - `destIsUnusual` — flag for suspicious/mule-like destination accounts.
   - `isOddHour` — flag for transactions between 12am–5am.
2. **Model**: Random Forest classifier (`class_weight="balanced_subsample"` to handle the class imbalance between fraud and legitimate transactions), inside a scikit-learn `Pipeline` with `StandardScaler` for numeric features and `OneHotEncoder` for transaction type.
3. **Evaluation**: stratified 75/25 train/test split, evaluated on ROC-AUC, PR-AUC (more meaningful than accuracy for imbalanced fraud data), precision, recall, and F1 for the fraud class.

## 4. Results (held-out test set)

| Metric | Score |
|---|---|
| ROC-AUC | 1.00 |
| PR-AUC | 1.00 |
| Precision (fraud class) | 0.99 |
| Recall (fraud class) | 1.00 |
| F1 (fraud class) | 1.00 |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[14998, 2], [0, 240]]`

See `models/feature_importance.png` and `models/pr_curve.png` for plots.
The strongest predictors were `drainRatioOrig`, `destIsUnusual`, and the
balance-consistency error features — consistent with real-world fraud
intuition (fraud tends to drain accounts and route through unfamiliar
destinations).

## 5. Project Structure

```
mobile-money-fraud-detection/
├── data/
│   └── mobile_money_transactions.csv   # generated synthetic dataset
├── models/
│   ├── fraud_model.pkl                 # trained pipeline
│   ├── metrics.json                    # evaluation results
│   ├── feature_importance.png
│   └── pr_curve.png
├── src/
│   ├── generate_data.py                # synthetic data generator
│   ├── train_model.py                  # feature engineering + training + evaluation
│   └── predict.py                      # CLI to score one transaction
├── app.py                              # Streamlit demo app
├── requirements.txt
└── README.md
```

## 6. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python src/generate_data.py

# 3. Train the model
python src/train_model.py

# 4. Score a transaction from the command line
python src/predict.py --type TRANSFER --amount 150000 \
    --old_orig 160000 --new_orig 10000 \
    --old_dest 0 --new_dest 150000 \
    --hour 2 --dest_name M900123

# 5. Launch the interactive demo app
streamlit run app.py
```

The Streamlit app lets you:
- Enter a transaction manually or load a random sample, and see the fraud probability with a plain-English explanation of the key risk signals.
- Upload a CSV of transactions to score in bulk and download the flagged results.

## 7. Limitations & Next Steps

- Trained on **synthetic** data — needs revalidation on real, labeled transactions before production use.
- Class imbalance handling could be extended with SMOTE or cost-sensitive thresholds tuned to the business's tolerance for false positives vs. missed fraud.
- Next steps: add device/behavioral features (device ID, location, transaction velocity per account), monitor for model drift, and set up a feedback loop where confirmed fraud/legit outcomes retrain the model periodically.

## 8. Author's Note

This is my individual capstone submission for the 3MTT NextGen Data
Science track (project DS-09 — Mobile-Money Fraud Detection), built as
an MVP demonstrating the full pipeline: data generation, feature
engineering, model training/evaluation, and an interactive demo.
