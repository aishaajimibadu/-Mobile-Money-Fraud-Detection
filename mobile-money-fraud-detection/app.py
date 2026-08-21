"""
app.py
------
Streamlit demo app for the Mobile-Money Fraud Detection capstone project.

Run:
    streamlit run app.py

Lets you enter (or randomly sample) a transaction and see the model's
fraud probability + explanation, or batch-score an uploaded CSV.
"""

import joblib
import pandas as pd
import streamlit as st

from src.train_model import engineer_features

st.set_page_config(page_title="Mobile-Money Fraud Detection", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    bundle = joblib.load("models/fraud_model.pkl")
    return bundle["pipeline"], bundle["numeric_features"], bundle["categorical_features"], bundle.get("model" , None)  
@st.cache_data
def load_sample_data():
    return pd.read_csv("data/mobile_money_transactions.csv")

pipeline, numeric_features, categorical_features = load_model()

st.title("🛡️ Mobile-Money Fraud Detection")
st.caption("3MTT NextGen Data Science Capstone — DS-09")

tab1, tab2 = st.tabs(["🔍 Score a single transaction", "📄 Batch score a CSV"])

with tab1:
    st.subheader("Enter transaction details")

    df_sample = load_sample_data()
    if st.button("🎲 Fill with a random sample transaction"):
        row = df_sample.sample(1).iloc[0]
        st.session_state.update({
            "type": row["type"], "amount": float(row["amount"]),
            "oldOrig": float(row["oldbalanceOrig"]), "newOrig": float(row["newbalanceOrig"]),
            "oldDest": float(row["oldbalanceDest"]), "newDest": float(row["newbalanceDest"]),
            "hour": int(row["hour_of_day"]), "dest_name": row["nameDest"],
        })

    col1, col2 = st.columns(2)
    with col1:
        txn_type = st.selectbox("Transaction type", ["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT", "AIRTIME"],
                                 index=["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT", "AIRTIME"].index(
                                     st.session_state.get("type", "TRANSFER")))
        amount = st.number_input("Amount (₦)", min_value=0.0, value=st.session_state.get("amount", 50000.0))
        hour = st.slider("Hour of day", 0, 23, st.session_state.get("hour", 12))
        dest_name = st.text_input("Destination account ID", st.session_state.get("dest_name", "C123456"))
    with col2:
        old_orig = st.number_input("Sender balance BEFORE (₦)", min_value=0.0, value=st.session_state.get("oldOrig", 60000.0))
        new_orig = st.number_input("Sender balance AFTER (₦)", min_value=0.0, value=st.session_state.get("newOrig", 10000.0))
        old_dest = st.number_input("Recipient balance BEFORE (₦)", min_value=0.0, value=st.session_state.get("oldDest", 0.0))
        new_dest = st.number_input("Recipient balance AFTER (₦)", min_value=0.0, value=st.session_state.get("newDest", 50000.0))

    if st.button("Check for fraud", type="primary"):
        row = pd.DataFrame([{
            "type": txn_type, "amount": amount, "hour_of_day": hour,
            "oldbalanceOrig": old_orig, "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
            "nameDest": dest_name,
        }])
        row = engineer_features(row)
        prob = pipeline.predict_proba(row[numeric_features + categorical_features])[0, 1]

        st.metric("Fraud probability", f"{prob*100:.1f}%")
        if prob >= 0.5:
            st.error("⚠️ Flagged as likely FRAUDULENT")
        else:
            st.success("✅ Looks legitimate")

        with st.expander("Why? (key signals)"):
            st.write(f"- Drain ratio (amount / sender balance): **{row['drainRatioOrig'].iloc[0]:.2f}**")
            st.write(f"- Account emptied after transaction: **{bool(row['accountEmptied'].iloc[0])}**")
            st.write(f"- Unusual destination pattern: **{bool(row['destIsUnusual'].iloc[0])}**")
            st.write(f"- Odd hour (12am-5am): **{bool(row['isOddHour'].iloc[0])}**")

with tab2:
    st.subheader("Upload a CSV of transactions to score in bulk")
    st.caption("Required columns: type, amount, oldbalanceOrig, newbalanceOrig, "
               "oldbalanceDest, newbalanceDest, nameDest, hour_of_day")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        batch_feat = engineer_features(batch)
        probs = pipeline.predict_proba(batch_feat[numeric_features + categorical_features])[:, 1]
        batch["fraud_probability"] = probs
        batch["flagged"] = probs >= 0.5
        st.write(f"Flagged **{batch['flagged'].sum()}** of {len(batch)} transactions as likely fraud.")
        st.dataframe(batch.sort_values("fraud_probability", ascending=False).head(50))
        st.download_button("Download scored results", batch.to_csv(index=False), "scored_transactions.csv")

st.divider()
st.caption("MVP model trained on synthetic mobile-money transaction data. "
           "See README.md for methodology, data caveats, and results.")
