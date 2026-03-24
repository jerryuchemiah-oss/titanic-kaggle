"""
Model training script for the Titanic Kaggle competition.

Usage:
    python3 src/train.py
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, skipping XGBClassifier.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed, skipping LGBMClassifier.")

# Allow running from repo root
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import engineer_features

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_models() -> dict:
    """Return dict of {name: estimator} for all candidate models."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, solver="lbfgs", C=0.5, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    return models


def print_feature_importances(model, feature_names: list, top_n: int = 15) -> None:
    """Print top feature importances if the model supports it."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Normalize to [0, 1] so the bar chart is sensible regardless of scale
        total = importances.sum()
        if total > 0:
            importances = importances / total
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        print("\n  Top feature importances (normalized):")
        for name, imp in pairs[:top_n]:
            bar = "#" * int(imp * 40)
            print(f"    {name:<20} {imp:.4f}  {bar}")
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_[0])
        pairs = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)
        print("\n  Top |coefficient| values:")
        for name, coef in pairs[:top_n]:
            print(f"    {name:<20} {coef:.4f}")


def main() -> None:
    print("=" * 60)
    print("Titanic — Model Training")
    print("=" * 60)

    # ── Load & engineer features ───────────────────────────────────────────────
    print(f"\nLoading data from: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {raw.shape}")

    df = engineer_features(raw, is_train=True)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    feature_names = list(X.columns)

    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Survival rate : {y.mean():.3f}")

    # ── Cross-validation ───────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()

    cv_results = {}
    print("\n--- 5-Fold Cross-Validation Scores ---")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_results[name] = scores
        print(
            f"  {name:<22} mean={scores.mean():.4f}  std={scores.std():.4f}"
            f"  folds={np.round(scores, 4).tolist()}"
        )

    # ── Select best model ──────────────────────────────────────────────────────
    best_name = max(cv_results, key=lambda n: cv_results[n].mean())
    best_score = cv_results[best_name].mean()
    print(f"\nBest model: {best_name}  (CV accuracy = {best_score:.4f})")

    # ── Retrain on full data ───────────────────────────────────────────────────
    best_model = models[best_name]
    print(f"\nRetraining {best_name} on full training set …")
    best_model.fit(X, y)

    # ── Feature importances ────────────────────────────────────────────────────
    print_feature_importances(best_model, feature_names)

    # ── Save model ─────────────────────────────────────────────────────────────
    payload = {
        "model": best_model,
        "model_name": best_name,
        "cv_score": best_score,
        "feature_names": feature_names,
        "cv_results": {k: v.tolist() for k, v in cv_results.items()},
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
