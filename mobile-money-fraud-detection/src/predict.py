"""
predict.py
----------
Command-line tool to score a single transaction using the trained model.

Example:
    python src/predict.py --type TRANSFER --amount 150000 \
        --old_orig 160000 --new_orig 10000 \
        --old_dest 0 --new_dest 150000 \
        --hour 2 --dest_name M900123
"""

import argparse
import joblib
import pandas as pd

from train_model import engineer_features


def main():
    parser = argparse.ArgumentParser(description="Score a mobile-money transaction for fraud risk.")
    parser.add_argument("--type", required=True, choices=["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT", "AIRTIME"])
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--old_orig", type=float, required=True, help="Sender balance before")
    parser.add_argument("--new_orig", type=float, required=True, help="Sender balance after")
    parser.add_argument("--old_dest", type=float, required=True, help="Recipient balance before")
    parser.add_argument("--new_dest", type=float, required=True, help="Recipient balance after")
    parser.add_argument("--hour", type=int, required=True, help="Hour of day (0-23)")
    parser.add_argument("--dest_name", type=str, default="C000000")
    parser.add_argument("--model_path", type=str, default="models/fraud_model.pkl")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    pipeline = bundle["pipeline"]
    numeric_features = bundle["numeric_features"]
    categorical_features = bundle["categorical_features"]

    row = pd.DataFrame([{
        "type": args.type, "amount": args.amount, "hour_of_day": args.hour,
        "oldbalanceOrig": args.old_orig, "newbalanceOrig": args.new_orig,
        "oldbalanceDest": args.old_dest, "newbalanceDest": args.new_dest,
        "nameDest": args.dest_name,
    }])
    row = engineer_features(row)
    prob = pipeline.predict_proba(row[numeric_features + categorical_features])[0, 1]

    verdict = "FRAUD" if prob >= 0.5 else "LEGITIMATE"
    print(f"Fraud probability: {prob*100:.2f}%")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
