"""
Streamlit demo app for the Credit Card Fraud Detection model.

Run locally with:
    streamlit run app/app.py

Lets you either:
  1. Upload a CSV of transactions and get fraud predictions for all of them, or
  2. Manually enter a single transaction's values and get an instant prediction.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, feature_names, metadata


model, scaler, feature_names, metadata = load_artifacts()

st.title("💳 Credit Card Fraud Detection")
st.caption(
    f"Model: **{metadata['best_model']}** · "
    f"Test PR-AUC: **{metadata['metrics'][0]['pr_auc']}%** · "
    f"Evaluated on a real-world imbalanced test set "
    f"({metadata['test_fraud_rate_pct']}% fraud rate)"
)

with st.expander("ℹ️ About this model"):
    st.write(
        "Trained on the ULB Kaggle Credit Card Fraud dataset (284,807 transactions, "
        "0.173% fraud). Unlike naive approaches that undersample the test set, this "
        "model is evaluated on the **original, imbalanced** distribution — the "
        "numbers below reflect realistic performance, not inflated ones."
    )
    st.dataframe(pd.DataFrame(metadata["metrics"]), use_container_width=True)

threshold = st.slider(
    "Fraud probability threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Transactions scoring above this probability are flagged as fraud. "
    "Lower it to catch more fraud at the cost of more false alarms.",
)

tab1, tab2 = st.tabs(["📄 Score a CSV file", "✍️ Enter a single transaction"])

with tab1:
    st.write("Upload a CSV with the same columns as the training data (Time, V1-V28, Amount).")
    uploaded = st.file_uploader("Upload transactions CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = set(feature_names) - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {sorted(missing)}")
        else:
            X = df[feature_names].copy()
            X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
            proba = model.predict_proba(X)[:, 1]
            pred = (proba >= threshold).astype(int)

            result = df.copy()
            result["fraud_probability"] = proba.round(4)
            result["predicted_class"] = pred

            n_flagged = pred.sum()
            st.success(f"Scored {len(result):,} transactions — {n_flagged} flagged as fraud.")
            st.dataframe(
                result.sort_values("fraud_probability", ascending=False),
                use_container_width=True,
            )
            st.download_button(
                "Download results as CSV",
                result.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )

with tab2:
    st.write(
        "The V1-V28 features are anonymized PCA components from the original dataset "
        "and aren't human-interpretable individually — this form is mainly useful for "
        "demoing the model with a real row copied from the dataset."
    )
    cols = st.columns(4)
    values = {}
    for i, feat in enumerate(feature_names):
        with cols[i % 4]:
            default = 0.0
            values[feat] = st.number_input(feat, value=default, format="%.4f", key=feat)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([values])[feature_names]
        row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])
        proba = model.predict_proba(row)[:, 1][0]
        pred = int(proba >= threshold)

        if pred == 1:
            st.error(f"🚨 Flagged as FRAUD — probability {proba:.2%}")
        else:
            st.success(f"✅ Looks legitimate — fraud probability {proba:.2%}")
