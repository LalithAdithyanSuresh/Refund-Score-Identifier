"""
train_classifier.py
────────────────────
Trains an XGBoost classifier (falls back to Logistic Regression) on the
synthetic training dataset and evaluates it.

Steps
─────
1. Load training_data.csv
2. Extract features for every complaint (Sentence-BERT + others)
3. Build combined feature matrix X and label vector y
4. 80/20 train-test split
5. Fit XGBoost (preferred) or Logistic Regression
6. Evaluate: accuracy, precision, recall, F1
7. Save model.pkl and features.csv
"""

import os
import csv
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
from feature_extractor import extract_features, warm_cache, ISSUE_TYPE_REFS

TRAINING_CSV = "training_data.csv"
FEATURES_CSV = "features.csv"
MODEL_PKL    = "model.pkl"


# ── 1. Load data ─────────────────────────────────────────────────────────────

def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "complaint_text" in df.columns, "Missing 'complaint_text' column"
    assert "label" in df.columns,          "Missing 'label' column"
    print(f"Loaded {len(df)} training samples from '{path}'.")
    print(f"  APPROVE (1): {(df['label']==1).sum()}")
    print(f"  REJECT  (0): {(df['label']==0).sum()}")
    return df


# ── 2. Feature extraction ─────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    For each row, call extract_features() and collect results.
    Returns X (ndarray), y (ndarray), rows (list of dicts for CSV).
    """
    X_rows   = []
    y_vals   = []
    csv_rows = []

    total = len(df)
    print(f"\nExtracting features for {total} samples...")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        text  = str(row["complaint_text"])
        label = int(row["label"])
        feats = extract_features(text)

        X_rows.append(feats["feature_vector"])
        y_vals.append(label)

        issue_name = ISSUE_TYPE_REFS[feats["issue_type"]].split()[0]
        csv_rows.append({
            "complaint_text": text,
            "label":          label,
            "severity":       feats["severity"],
            "sentiment":      feats["sentiment"],
            "urgency":        feats["urgency"],
            "issue_type":     feats["issue_type"],
            "issue_name":     issue_name,
        })

        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} processed...")

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=int)
    return X, y, csv_rows


# ── 3. Model selection & training ─────────────────────────────────────────────

def get_classifier():
    """Return XGBoost if available, else Logistic Regression."""
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        print("\nUsing classifier: XGBoost")
        return clf, "XGBoost"
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        print("\nXGBoost not found. Using classifier: Logistic Regression")
        return clf, "LogisticRegression"


def train_model(X: np.ndarray, y: np.ndarray):
    """Split, train, evaluate, return (model, metrics_dict)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

    clf, clf_name = get_classifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 55)
    print(f"   CLASSIFIER : {clf_name}")
    print("=" * 55)
    print(f"   Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   Precision  : {prec:.4f}")
    print(f"   Recall     : {rec:.4f}")
    print(f"   F1-Score   : {f1:.4f}")
    print("=" * 55)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["REJECT(0)", "APPROVE(1)"]))

    metrics = {"classifier": clf_name, "accuracy": round(acc, 4),
               "precision": round(prec, 4), "recall": round(rec, 4),
               "f1": round(f1, 4)}
    return clf, metrics


# ── 4. Save outputs ───────────────────────────────────────────────────────────

def save_features_csv(csv_rows: list[dict], path: str):
    if not csv_rows:
        return
    fieldnames = list(csv_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nFeatures saved to '{path}'  ({len(csv_rows)} rows)")


def save_model(clf, path: str):
    joblib.dump(clf, path)
    print(f"Model saved to '{path}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   REFUND CLASSIFIER — TRAINING")
    print("=" * 55)

    # Warm Sentence-BERT cache
    warm_cache()

    # Load data
    df = load_training_data(TRAINING_CSV)

    # Extract features
    X, y, csv_rows = build_feature_matrix(df)
    print(f"\nFeature matrix shape: {X.shape}")

    # Train
    clf, metrics = train_model(X, y)

    # Save
    save_features_csv(csv_rows, FEATURES_CSV)
    save_model(clf, MODEL_PKL)

    print("\nAll outputs saved:")
    print(f"  {FEATURES_CSV}")
    print(f"  {MODEL_PKL}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
