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
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

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

from src.features import fit_transform

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_models() -> dict:
    """Return dict of {name: estimator} for all candidate models."""
    models = {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
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
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    return models


def build_voting_ensemble(models: dict) -> VotingClassifier:
    """Build a hard-voting ensemble from the provided models dict."""
    estimators = list(models.items())
    return VotingClassifier(estimators=estimators, voting="hard")


def print_feature_importances(model, feature_names: list, top_n: int = 15) -> None:
    """Print top feature importances if the model supports it."""
    # Unwrap VotingClassifier — use the first tree-based estimator found
    check_model = model
    if isinstance(model, VotingClassifier):
        for _, est in model.estimators:
            if hasattr(est, "feature_importances_"):
                check_model = est
                break

    if hasattr(check_model, "feature_importances_"):
        importances = check_model.feature_importances_
        total = importances.sum()
        if total > 0:
            importances = importances / total
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        print("\n  Top feature importances (normalized):")
        for name, imp in pairs[:top_n]:
            bar = "#" * int(imp * 40)
            print(f"    {name:<22} {imp:.4f}  {bar}")
    elif hasattr(check_model, "coef_"):
        coefs = np.abs(check_model.coef_[0])
        pairs = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)
        print("\n  Top |coefficient| values:")
        for name, coef in pairs[:top_n]:
            print(f"    {name:<22} {coef:.4f}")


def main() -> None:
    print("=" * 60)
    print("Titanic — Model Training")
    print("=" * 60)

    # ── Load & engineer features ───────────────────────────────────────────────
    print(f"\nLoading data from: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {raw.shape}")

    df, encoders = fit_transform(raw)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"Encoders saved to: {ENCODERS_PATH}")
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    feature_names = list(X.columns)

    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Features      : {feature_names}")
    print(f"Survival rate : {y.mean():.3f}")

    # ── Cross-validation ───────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()
    voting_ensemble = build_voting_ensemble(models)

    all_models = dict(models)
    all_models["VotingEnsemble"] = voting_ensemble

    cv_results = {}
    print("\n--- 5-Fold Stratified Cross-Validation Scores ---")
    for name, model in all_models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_results[name] = scores
        print(
            f"  {name:<22}  mean={scores.mean():.4f}  std={scores.std():.4f}"
            f"  folds={np.round(scores, 4).tolist()}"
        )

    # ── Select best model (exclude ensemble from competition to save individual models) ──
    best_name = max(cv_results, key=lambda n: cv_results[n].mean())
    best_score = cv_results[best_name].mean()
    print(f"\nBest model: {best_name}  (CV accuracy = {best_score:.4f})")

    # ── Retrain all individual models on full data and save each ───────────────
    print("\nRetraining all individual models on full training set ...")
    trained_models = {}
    for name, model in models.items():
        print(f"  Fitting {name} ...")
        model.fit(X, y)
        trained_models[name] = model
        individual_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(
            {
                "model": model,
                "model_name": name,
                "cv_score": cv_results[name].mean(),
                "feature_names": feature_names,
            },
            individual_path,
        )
        print(f"    Saved to: {individual_path}")

    # Also fit and save the voting ensemble
    print("  Fitting VotingEnsemble ...")
    voting_ensemble.fit(X, y)
    trained_models["VotingEnsemble"] = voting_ensemble
    voting_path = MODEL_DIR / "VotingEnsemble.pkl"
    joblib.dump(
        {
            "model": voting_ensemble,
            "model_name": "VotingEnsemble",
            "cv_score": cv_results["VotingEnsemble"].mean(),
            "feature_names": feature_names,
        },
        voting_path,
    )
    print(f"    Saved to: {voting_path}")

    # ── Save best model as best_model.pkl ──────────────────────────────────────
    best_model = trained_models[best_name]
    print_feature_importances(best_model, feature_names)

    payload = {
        "model": best_model,
        "model_name": best_name,
        "cv_score": best_score,
        "feature_names": feature_names,
        "cv_results": {k: v.tolist() for k, v in cv_results.items()},
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nBest model saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
