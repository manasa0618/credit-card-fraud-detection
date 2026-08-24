"""
Credit Card Fraud Detection — Training Pipeline
=================================================
Trains and compares multiple classifiers for fraud detection on the
ULB Credit Card Fraud dataset, using techniques appropriate for a
severely imbalanced classification problem:

  - Stratified train/test split BEFORE any resampling, so the test
    set reflects the real-world (highly imbalanced) distribution.
  - Feature scaling for 'Time' and 'Amount' (V1-V28 are already PCA
    components on a similar scale).
  - Three imbalance-handling strategies compared side by side:
      1. class_weight='balanced' (Logistic Regression, Random Forest)
      2. SMOTE oversampling applied ONLY to the training fold
      3. XGBoost with scale_pos_weight
  - Evaluation on the untouched imbalanced test set using metrics
    that are meaningful for rare-event detection: ROC-AUC, PR-AUC
    (average precision), F1, precision, recall, and a confusion
    matrix — NOT plain accuracy, which is meaningless when 99.83%
    of transactions are legitimate.
  - 5-fold stratified cross-validation (scored on PR-AUC) for the
    best-performing model, to sanity-check the single-split result.
  - Saves the best model + scaler to models/ for reuse in the
    Streamlit app and predict.py.

Usage:
    python src/train.py
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
IMAGES_DIR = ROOT / "images"
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    return df


def prepare_features(df: pd.DataFrame):
    """Scale Time and Amount; V1-V28 are already PCA components on a
    comparable scale so they're left as-is."""
    df = df.copy()
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
    X = df.drop(columns="Class")
    y = df["Class"]
    return X, y, scaler


def evaluate(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }
    results.append(metrics)

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}   PR-AUC: {metrics['pr_auc']:.4f}")
    return y_pred, y_proba, metrics


def plot_confusion_matrices(fitted_models, X_test, y_test):
    fig, axes = plt.subplots(1, len(fitted_models), figsize=(6 * len(fitted_models), 5))
    if len(fitted_models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, fitted_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrices.png", dpi=150)
    plt.close()


def plot_pr_roc_curves(fitted_models, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name, model in fitted_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, name=name, ax=axes[0])
        RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=axes[1])
    axes[0].set_title("Precision-Recall Curve (imbalanced test set)")
    axes[1].set_title("ROC Curve (imbalanced test set)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "pr_roc_curves.png", dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )[:15]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, color="steelblue")
    plt.title(f"Top 15 Feature Importances — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main():
    df = load_data()
    X, y, scaler = prepare_features(df)

    # Split BEFORE any resampling — test set stays realistically imbalanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean()*100:.3f}%  Test fraud rate: {y_test.mean()*100:.3f}%")

    # SMOTE — training data only, never touches the test set
    print("\nApplying SMOTE to training data only...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {y_train_smote.value_counts().to_dict()}")

    results = []
    fitted_models = {}

    # 1. Logistic Regression with class_weight balanced (no resampling needed)
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    evaluate("Logistic Regression (class_weight)", lr, X_test, y_test, results)
    fitted_models["Logistic Regression"] = lr
    print(f"  trained in {time.time()-t0:.1f}s")

    # 2. Decision Tree trained on SMOTE-resampled data
    t0 = time.time()
    dt = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    dt.fit(X_train_smote, y_train_smote)
    evaluate("Decision Tree (SMOTE)", dt, X_test, y_test, results)
    fitted_models["Decision Tree"] = dt
    print(f"  trained in {time.time()-t0:.1f}s")

    # 3. Random Forest with class_weight balanced
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    evaluate("Random Forest (class_weight)", rf, X_test, y_test, results)
    fitted_models["Random Forest"] = rf
    print(f"  trained in {time.time()-t0:.1f}s")

    # 4. XGBoost with scale_pos_weight
    t0 = time.time()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    evaluate("XGBoost (scale_pos_weight)", xgb, X_test, y_test, results)
    fitted_models["XGBoost"] = xgb
    print(f"  trained in {time.time()-t0:.1f}s")

    # Results table
    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    results_df_display = results_df.copy()
    for col in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        results_df_display[col] = (results_df_display[col] * 100).round(2)
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by PR-AUC — the right metric for rare-event detection)")
    print("=" * 60)
    print(results_df_display.to_string(index=False))
    results_df_display.to_csv(ROOT / "results.csv", index=False)

    # Best model = highest PR-AUC (not accuracy — accuracy is ~99.8% for
    # a model that predicts "legit" every time, so it's useless here)
    best_name = results_df.iloc[0]["model"]
    best_key = [k for k in fitted_models if k in best_name][0]
    best_model = fitted_models[best_key]
    print(f"\nBest model by PR-AUC: {best_name}")

    # Cross-validation sanity check for the best model on the original
    # (unresampled) training data, scored on PR-AUC
    print("\nRunning 5-fold stratified cross-validation on best model...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if best_key == "Decision Tree":
        cv_scores = cross_val_score(
            DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
            X_train_smote, y_train_smote, cv=cv, scoring="average_precision", n_jobs=-1,
        )
    else:
        cv_scores = cross_val_score(
            best_model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
        )
    print(f"CV PR-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Plots
    plot_confusion_matrices(fitted_models, X_test, y_test)
    plot_pr_roc_curves(fitted_models, X_test, y_test)
    plot_feature_importance(best_model, X.columns, best_name)

    # Save best model + scaler for the Streamlit app / predict.py
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(list(X.columns), MODELS_DIR / "feature_names.pkl")

    metadata = {
        "best_model": best_name,
        "test_set_size": len(X_test),
        "test_fraud_rate_pct": round(y_test.mean() * 100, 4),
        "cv_pr_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_pr_auc_std": round(float(cv_scores.std()), 4),
        "metrics": results_df_display.to_dict(orient="records"),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model ({best_name}) to {MODELS_DIR}/best_model.pkl")
    print(f"Saved plots to {IMAGES_DIR}/")
    print(f"Saved results table to {ROOT}/results.csv")


if __name__ == "__main__":
    main()
