"""
Feature engineering pipeline for the Titanic Kaggle competition.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def extract_title(name: str) -> str:
    """Extract and normalize title from passenger name."""
    title = name.split(",")[1].split(".")[0].strip()
    rare_titles = {
        "Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona",
    }
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    if title in title_map:
        return title_map[title]
    if title in rare_titles:
        return "Rare"
    return title


def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (train or test).
    is_train : bool
        If True, PassengerId is dropped along with other ID-like columns.
        If False (test set), PassengerId is retained for submission assembly.

    Returns
    -------
    pd.DataFrame
        Clean, model-ready dataframe.
    """
    df = df.copy()

    # --- PassengerId ---
    passenger_ids = df["PassengerId"].copy() if not is_train else None

    # --- Title ---
    df["Title"] = df["Name"].apply(extract_title)

    # --- Missing Age: fill with median grouped by Pclass + Sex ---
    age_medians = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_medians)
    # Fallback for any remaining NaN (e.g. group is entirely NaN)
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # --- Missing Embarked: fill with mode ---
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # --- Missing Fare: fill with median ---
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # --- FamilySize ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # --- IsAlone ---
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # --- AgeGroup ---
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 120],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    )

    # --- FareBin ---
    df["FareBin"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "VeryHigh"])

    # --- Encode categoricals ---
    le = LabelEncoder()
    for col in ["Sex", "Embarked", "Title", "AgeGroup", "FareBin"]:
        df[col] = le.fit_transform(df[col].astype(str))

    # --- Drop columns not useful for modelling ---
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if not is_train:
        # Reattach PassengerId for later submission assembly
        df["PassengerId"] = passenger_ids.values

    return df
