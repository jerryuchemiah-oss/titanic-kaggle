"""
Prediction / submission generation script for the Titanic Kaggle competition.

Usage:
    python3 src/predict.py
"""

import sys
import pathlib
import datetime

import pandas as pd
import joblib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import engineer_features

# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
SUBMISSION_DIR = ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("Titanic — Prediction / Submission Generation")
    print("=" * 60)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model from: {MODEL_PATH}")
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    model_name = payload["model_name"]
    cv_score = payload["cv_score"]
    feature_names = payload["feature_names"]

    print(f"Model      : {model_name}")
    print(f"CV Accuracy: {cv_score:.4f}")

    # ── Load & process test data ───────────────────────────────────────────────
    print(f"\nLoading test data from: {TEST_PATH}")
    raw_test = pd.read_csv(TEST_PATH)
    print(f"Test shape : {raw_test.shape}")

    df_test = engineer_features(raw_test, is_train=False)

    # Separate PassengerId before prediction
    passenger_ids = df_test["PassengerId"]
    X_test = df_test.drop(columns=["PassengerId"])

    # Align columns to training feature set
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_names]

    print(f"Feature matrix: {X_test.shape[0]} samples × {X_test.shape[1]} features")

    # ── Generate predictions ───────────────────────────────────────────────────
    predictions = model.predict(X_test)
    print(f"\nPredicted survivors: {predictions.sum()} / {len(predictions)}"
          f"  ({predictions.mean():.2%})")

    # ── Build submission DataFrame ─────────────────────────────────────────────
    submission = pd.DataFrame({
        "PassengerId": passenger_ids.values,
        "Survived": predictions.astype(int),
    })

    # ── Save submission ────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = SUBMISSION_DIR / f"submission_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to: {submission_path}")
    print(submission.head(10).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
