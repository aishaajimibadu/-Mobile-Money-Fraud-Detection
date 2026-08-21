"""
train_model.py
---------------
Feature engineering + model training/evaluation for the
Mobile-Money Fraud Detection capstone.

Run:
    python src/train_model.py

Outputs:
    models/fraud_model.pkl   - trained pipeline (preprocessing + classifier)
    models/metrics.json      - evaluation metrics on the held-out test set
    models/feature_importance.png
    models/pr_curve.png
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix
)

DATA_PATH = "data/mobile_money_transactions.csv"
MODEL_PATH = "models/fraud_model.pkl"
METRICS_PATH = "models/metrics.json"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features that help separate fraud from legitimate mobile-money
    transactions. These mirror signals commonly used in real fraud systems:
    balance-consistency errors, drain ratio, and origin/destination patterns.
    """
    df = df.copy()

    # Balance-consistency errors: legitimate transactions should roughly
    # satisfy old_balance - amount = new_balance. Fraudulent/edge transactions
    # often break this, or exactly zero the account out.
    df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrig"]).abs()
    df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).abs()

    # Ratio of the transaction amount to the sender's prior balance.
    # A value close to 1 means the account was drained almost completely.
    df["drainRatioOrig"] = df["amount"] / (df["oldbalanceOrig"] + 1)
    df["accountEmptied"] = (df["newbalanceOrig"] <= 1).astype(int)

    # Destination is a "mule-like" id if it starts with M (synthetic flag
    # standing in for e.g. newly-created / unverified recipient wallets).
    df["destIsUnusual"] = df["nameDest"].str.startswith("M").astype(int)

    # Odd-hour flag: transactions between midnight and 5am are less common
    # for normal usage and correlate with account-takeover fraud.
    df["isOddHour"] = df["hour_of_day"].between(0, 5).astype(int)

    return df


def build_pipeline(numeric_features, categorical_features):
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def main():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    numeric_features = [
        "amount", "oldbalanceOrig", "newbalanceOrig", "oldbalanceDest",
        "newbalanceDest", "errorBalanceOrig", "errorBalanceDest",
        "drainRatioOrig", "accountEmptied", "destIsUnusual", "isOddHour",
        "hour_of_day",
    ]
    categorical_features = ["type"]

    X = df[numeric_features + categorical_features]
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_fraud_class": report["1"]["precision"],
        "recall_fraud_class": report["1"]["recall"],
        "f1_fraud_class": report["1"]["f1-score"],
        "confusion_matrix": cm,
        "n_test": len(y_test),
        "n_fraud_test": int(y_test.sum()),
    }

    print("=== Evaluation on held-out test set ===")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"Precision (fraud): {metrics['precision_fraud_class']:.3f}")
    print(f"Recall (fraud):    {metrics['recall_fraud_class']:.3f}")
    print(f"F1 (fraud):        {metrics['f1_fraud_class']:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:", cm)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({
        "pipeline": pipeline,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }, MODEL_PATH)
    print(f"\nSaved trained pipeline -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")

    # Feature importance plot
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_names
    importances = pipeline.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1][:12]

    plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#2f855a")
    plt.title("Top Feature Importances - Fraud Detection Model")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=150)
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#2b6cb0")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (PR-AUC = {pr_auc:.3f})")
    plt.tight_layout()
    plt.savefig("models/pr_curve.png", dpi=150)
    plt.close()

    print("Saved plots -> models/feature_importance.png, models/pr_curve.png")


if __name__ == "__main__":
    main()
