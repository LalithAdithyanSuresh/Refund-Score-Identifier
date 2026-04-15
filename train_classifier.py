"""
train_classifier.py
────────────────────
Trains an XGBoost classifier for multiple embedding techniques:
- Word2Vec
- GloVe
- FastText
- BERT
- InferSent
- SBERT

For each method, it:
1. Extracts features from training_data.csv
2. Trains an XGBoost classifier
3. Evaluates performance
4. Saves model_<name>.pkl, features_<name>.csv, and predictions_<name>.csv
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
from feature_extractor import (extract_features, warm_cache, ISSUE_TYPE_REFS, 
                               set_embedding_method)

TRAINING_CSV = "training_data.csv"
EMBEDDING_METHODS = ["word2vec", "glove", "fasttext", "bert", "infersent", "sbert"]


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "complaint_text" in df.columns, "Missing 'complaint_text' column"
    assert "label" in df.columns,          "Missing 'label' column"
    print(f"Loaded {len(df)} training samples from '{path}'.")
    return df


def build_feature_matrix(df: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X_rows   = []
    y_vals   = []
    csv_rows = []

    total = len(df)
    print(f"\nExtracting features using [{method}] for {total} samples...")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        text  = str(row["complaint_text"])
        label = int(row["label"])
        feats = extract_features(text, method=method)

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

        if i % 100 == 0 or i == total:
            print(f"  {i}/{total} processed...")

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=int)
    return X, y, csv_rows


def get_classifier():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


def train_and_eval(X: np.ndarray, y: np.ndarray, method: str, df_text: pd.Series):
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, range(len(X)), test_size=0.2, random_state=42, stratify=y
    )
    
    clf = get_classifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nResults for {method.upper()}:")
    print(f"  Accuracy : {acc:.4f} | F1: {f1:.4f}")

    metrics = {
        "embedding": method,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4)
    }

    # Prepare predictions CSV data (only for the test set)
    pred_rows = []
    test_texts = df_text.iloc[idx_test].values
    for text, actual, pred in zip(test_texts, y_test, y_pred):
        pred_rows.append({
            "complaint_text": text,
            "actual": actual,
            "predicted": pred,
            "status": "CORRECT" if actual == pred else "INCORRECT"
        })

    return clf, metrics, pred_rows


def save_csv(rows, path):
    if not rows: return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 60)
    print("   REFUND CLASSIFIER — MULTI-PIPELINE TRAINING")
    print("=" * 60)

    df = load_training_data(TRAINING_CSV)
    all_metrics = []

    for method in EMBEDDING_METHODS:
        print(f"\n>>> Starting Pipeline: {method.upper()}")
        try:
            # Set method and warm cache
            set_embedding_method(method)
            warm_cache(method)

            # Build features
            X, y, feat_rows = build_feature_matrix(df, method)
            
            # Train and evaluate
            clf, metrics, pred_rows = train_and_eval(X, y, method, df["complaint_text"])
            all_metrics.append(metrics)

            # Save artifacts
            save_model_path = f"model_{method}.pkl"
            save_feat_path = f"features_{method}.csv"
            save_pred_path = f"predictions_{method}.csv"

            joblib.dump(clf, save_model_path)
            save_csv(feat_rows, save_feat_path)
            save_csv(pred_rows, save_pred_path)

            print(f"  Artifacts saved for {method}")

        except Exception as e:
            print(f"  [ERROR] Pipeline {method} failed: {e}")

    # Save summary metrics
    with open("all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    print("\n" + "=" * 60)
    print("   FINAL COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Embedding':<12} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['embedding']:<12} | {m['accuracy']:<6.4f} | {m['precision']:<6.4f} | {m['recall']:<6.4f} | {m['f1']:<6.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
