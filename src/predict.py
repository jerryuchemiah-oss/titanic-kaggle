"""
Prediction / submission generation script for the Titanic Kaggle competition.

Uses soft probability blending across all saved models for the final prediction.

Usage:
    python3 src/predict.py
"""

import sys
import pathlib
import datetime

import numpy as np
import pandas as pd
import joblib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import transform

# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
MODEL_DIR = ROOT / "models"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"
SUBMISSION_DIR = ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Individual model files to blend (VotingEnsemble uses hard voting; skip it for soft blend)
INDIVIDUAL_MODEL_FILES = [
    "LogisticRegression.pkl",
    "RandomForest.pkl",
    "GradientBoosting.pkl",
    "XGBoost.pkl",
    "LightGBM.pkl",
]


def main() -> None:
    print("=" * 60)
    print("Titanic — Soft-Voting Ensemble Prediction")
    print("=" * 60)

    # ── Load & process test data ───────────────────────────────────────────────
    print(f"\nLoading test data from: {TEST_PATH}")
    raw_test = pd.read_csv(TEST_PATH)
    print(f"Test shape : {raw_test.shape}")

    encoders = joblib.load(ENCODERS_PATH)
    df_test = transform(raw_test, encoders)
    passenger_ids = df_test["PassengerId"]
    X_test_full = df_test.drop(columns=["PassengerId"])

    # ── Load models and collect probability predictions ────────────────────────
    prob_arrays = []
    loaded_model_names = []

    for fname in INDIVIDUAL_MODEL_FILES:
        fpath = MODEL_DIR / fname
        if not fpath.exists():
            print(f"  SKIP (not found): {fpath}")
            continue

        payload = joblib.load(fpath)
        model = payload["model"]
        model_name = payload["model_name"]
        feature_names = payload["feature_names"]
        cv_score = payload.get("cv_score", float("nan"))

        # Align columns to this model's training feature set
        X_test = X_test_full.copy()
        for col in feature_names:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[feature_names]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            prob_arrays.append(probs)
            loaded_model_names.append(model_name)
            print(f"  Loaded {model_name:<22}  CV={cv_score:.4f}  (predict_proba)")
        else:
            # Fallback: use hard predictions as 0/1 probabilities
            preds = model.predict(X_test).astype(float)
            prob_arrays.append(preds)
            loaded_model_names.append(model_name)
            print(f"  Loaded {model_name:<22}  CV={cv_score:.4f}  (predict, no proba)")

    if not prob_arrays:
        raise RuntimeError(
            "No model files found in models/. Run src/train.py first."
        )

    print(f"\nBlending {len(prob_arrays)} model(s): {loaded_model_names}")

    # ── Soft-vote: average probabilities, threshold at 0.5 ────────────────────
    avg_probs = np.mean(np.stack(prob_arrays, axis=0), axis=0)
    predictions = (avg_probs >= 0.5).astype(int)

    print(f"\nAvg probability stats:  min={avg_probs.min():.3f}  "
          f"max={avg_probs.max():.3f}  mean={avg_probs.mean():.3f}")
    print(f"Predicted survivors: {predictions.sum()} / {len(predictions)}"
          f"  ({predictions.mean():.2%})")

    # ── Build submission DataFrame ─────────────────────────────────────────────
    submission = pd.DataFrame({
        "PassengerId": passenger_ids.values,
        "Survived": predictions,
    })

    # ── Save submission ────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = SUBMISSION_DIR / f"submission_v3_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to: {submission_path}")
    print(submission.head(10).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
