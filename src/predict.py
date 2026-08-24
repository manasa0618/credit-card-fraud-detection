"""
Run predictions with the saved model on new transaction data.

Usage:
    python src/predict.py --input path/to/transactions.csv --output predictions.csv

Input CSV must contain the same columns as the training data (Time, V1-V28,
Amount) — no 'Class' column needed.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return model, scaler, feature_names


def predict(df: pd.DataFrame, model, scaler, feature_names, threshold: float = 0.5):
    df = df.copy()
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    X = df[feature_names]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of transactions to score")
    parser.add_argument("--output", default="predictions.csv", help="Where to write results")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold above which a transaction is flagged as fraud",
    )
    args = parser.parse_args()

    model, scaler, feature_names = load_artifacts()
    df = pd.read_csv(args.input)
    pred, proba = predict(df, model, scaler, feature_names, args.threshold)

    out = df.copy()
    out["fraud_probability"] = proba
    out["predicted_class"] = pred
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out)} transactions -> {args.output}")
    print(f"Flagged {pred.sum()} as fraud (threshold={args.threshold})")


if __name__ == "__main__":
    main()
