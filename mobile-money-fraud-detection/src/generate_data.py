"""
generate_data.py
-----------------
Generates a synthetic mobile-money transaction dataset for the
Mobile-Money Fraud Detection capstone project (3MTT NextGen - Data Science).

Why synthetic data?
Real mobile-money transaction logs (OPay, Paga, MTN MoMo, etc.) are private
and cannot be shared publicly. This script builds a realistic dataset that
mirrors the structure and fraud patterns seen in real mobile-money systems
(inspired by the publicly documented PaySim simulator), so the whole
pipeline - EDA, feature engineering, modelling, evaluation - can be
demonstrated end-to-end on data with a similar shape/behaviour to production
data.

Transaction types simulated: CASH_IN, CASH_OUT, TRANSFER, PAYMENT, AIRTIME

Fraud patterns injected (roughly modelled on real fraud typologies):
1. Account takeover  - a single large TRANSFER/CASH_OUT that drains an
   account shortly after a period of normal, low activity.
2. Mule/agent collusion - a TRANSFER to an account that is immediately
   followed by a CASH_OUT that empties it (money laundered out fast).
3. Rapid micro-fraud burst - several small TRANSFERs in a short time window
   after odd hours (SIM-swap-like behaviour).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_CUSTOMERS = 4000
N_AGENTS = 300
N_TRANSACTIONS = 60000
FRAUD_RATE_TARGET = 0.012  # ~1.2% fraud, similar order of magnitude to real mobile-money fraud rates

TXN_TYPES = ["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT", "AIRTIME"]
TXN_TYPE_WEIGHTS = [0.22, 0.28, 0.20, 0.20, 0.10]


def make_ids():
    customers = [f"C{100000+i}" for i in range(N_CUSTOMERS)]
    agents = [f"A{100000+i}" for i in range(N_AGENTS)]
    return customers, agents


def simulate_normal_transactions(customers, agents, n):
    steps = RNG.integers(0, 24 * 30, size=n)  # 30-day simulation, hourly steps
    types = RNG.choice(TXN_TYPES, size=n, p=TXN_TYPE_WEIGHTS)

    # Amount distributions differ by transaction type (lognormal, capped)
    amounts = np.zeros(n)
    for t, (mu, sigma, cap) in {
        "CASH_IN": (8.5, 1.0, 500000),
        "CASH_OUT": (8.3, 1.1, 500000),
        "TRANSFER": (8.0, 1.3, 800000),
        "PAYMENT": (7.2, 1.0, 150000),
        "AIRTIME": (5.0, 0.6, 5000),
    }.items():
        mask = types == t
        vals = RNG.lognormal(mean=mu, sigma=sigma, size=mask.sum())
        amounts[mask] = np.clip(vals, 50, cap)

    name_orig = RNG.choice(customers, size=n)
    # destination: agents for CASH_IN/CASH_OUT, customers for TRANSFER/PAYMENT/AIRTIME
    name_dest = np.empty(n, dtype=object)
    for i, t in enumerate(types):
        if t in ("CASH_IN", "CASH_OUT"):
            name_dest[i] = RNG.choice(agents)
        else:
            name_dest[i] = RNG.choice(customers)

    old_bal_orig = np.clip(RNG.lognormal(mean=9.0, sigma=1.2, size=n), 0, 2_000_000)
    # balance change roughly matches txn direction, with noise
    delta = np.where(np.isin(types, ["CASH_IN"]), amounts, -amounts)
    new_bal_orig = np.clip(old_bal_orig + delta + RNG.normal(0, 50, n), 0, None)

    old_bal_dest = np.clip(RNG.lognormal(mean=8.5, sigma=1.3, size=n), 0, 2_000_000)
    delta_dest = np.where(np.isin(types, ["CASH_OUT"]), -amounts, amounts)
    new_bal_dest = np.clip(old_bal_dest + delta_dest + RNG.normal(0, 50, n), 0, None)

    df = pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts.round(2),
        "nameOrig": name_orig,
        "oldbalanceOrig": old_bal_orig.round(2),
        "newbalanceOrig": new_bal_orig.round(2),
        "nameDest": name_dest,
        "oldbalanceDest": old_bal_dest.round(2),
        "newbalanceDest": new_bal_dest.round(2),
        "isFraud": 0,
    })
    return df


def inject_fraud(df, customers):
    n_fraud = int(len(df) * FRAUD_RATE_TARGET)
    fraud_rows = []

    # Pattern 1: account takeover (large drain, odd hour)
    for _ in range(n_fraud // 3):
        cust = RNG.choice(customers)
        old_bal = float(np.clip(RNG.lognormal(9.5, 1.0), 5000, 2_000_000))
        amount = old_bal * RNG.uniform(0.85, 1.0)  # drains almost everything
        step = int(RNG.choice(range(0, 24 * 30)) - RNG.choice(range(0, 24 * 30)) % 24 + RNG.integers(0, 5))
        step = max(step, 0) % (24 * 30)
        fraud_rows.append({
            "step": step,
            "type": RNG.choice(["TRANSFER", "CASH_OUT"]),
            "amount": round(amount, 2),
            "nameOrig": cust,
            "oldbalanceOrig": round(old_bal, 2),
            "newbalanceOrig": round(old_bal - amount, 2),
            "nameDest": f"M{RNG.integers(900000,999999)}",  # unusual/mule-like dest id
            "oldbalanceDest": 0.0,
            "newbalanceDest": round(amount, 2),
            "isFraud": 1,
        })

    # Pattern 2: mule collusion - TRANSFER immediately drained via CASH_OUT
    for _ in range(n_fraud // 3):
        cust = RNG.choice(customers)
        mule = f"M{RNG.integers(900000,999999)}"
        amount = float(np.clip(RNG.lognormal(9.0, 0.8), 5000, 600000))
        step = int(RNG.integers(0, 24 * 30 - 1))
        old_bal = amount + RNG.uniform(0, 5000)
        fraud_rows.append({
            "step": step, "type": "TRANSFER", "amount": round(amount, 2),
            "nameOrig": cust, "oldbalanceOrig": round(old_bal, 2),
            "newbalanceOrig": round(old_bal - amount, 2),
            "nameDest": mule, "oldbalanceDest": 0.0, "newbalanceDest": round(amount, 2),
            "isFraud": 1,
        })
        fraud_rows.append({
            "step": step + 1, "type": "CASH_OUT", "amount": round(amount * RNG.uniform(0.9, 1.0), 2),
            "nameOrig": mule, "oldbalanceOrig": round(amount, 2),
            "newbalanceOrig": round(amount * 0.03, 2),
            "nameDest": f"A{RNG.integers(100000,100299)}", "oldbalanceDest": round(RNG.uniform(0, 20000), 2),
            "newbalanceDest": round(amount * 0.9, 2),
            "isFraud": 1,
        })

    # Pattern 3: rapid micro-fraud burst (SIM-swap-like)
    n_bursts = max(1, n_fraud // 3 // 4)
    for _ in range(n_bursts):
        cust = RNG.choice(customers)
        base_step = int(RNG.integers(0, 24 * 30 - 5))
        old_bal = float(np.clip(RNG.lognormal(8.5, 1.0), 3000, 300000))
        bal = old_bal
        for k in range(4):
            amt = float(np.clip(RNG.lognormal(7.5, 0.7), 500, bal * 0.6 if bal > 500 else 500))
            new_bal = max(bal - amt, 0)
            fraud_rows.append({
                "step": base_step + k, "type": "TRANSFER", "amount": round(amt, 2),
                "nameOrig": cust, "oldbalanceOrig": round(bal, 2),
                "newbalanceOrig": round(new_bal, 2),
                "nameDest": f"M{RNG.integers(900000,999999)}",
                "oldbalanceDest": 0.0, "newbalanceDest": round(amt, 2),
                "isFraud": 1,
            })
            bal = new_bal

    fraud_df = pd.DataFrame(fraud_rows)
    combined = pd.concat([df, fraud_df], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return combined


def main():
    customers, agents = make_ids()
    normal_df = simulate_normal_transactions(customers, agents, N_TRANSACTIONS)
    full_df = inject_fraud(normal_df, customers)

    full_df["hour_of_day"] = full_df["step"] % 24
    full_df["day"] = full_df["step"] // 24

    out_path = "data/mobile_money_transactions.csv"
    full_df.to_csv(out_path, index=False)

    print(f"Generated {len(full_df):,} transactions -> {out_path}")
    print(f"Fraud cases: {full_df['isFraud'].sum():,} ({full_df['isFraud'].mean()*100:.2f}%)")
    print(full_df['type'].value_counts())


if __name__ == "__main__":
    main()
