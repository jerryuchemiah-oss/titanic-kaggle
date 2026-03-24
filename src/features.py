"""
Feature engineering pipeline for the Titanic Kaggle competition.

Encoders are fitted on training data and must be passed to transform test data,
ensuring train/test encoding consistency.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

_TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}

_ROYAL_TITLES = {"Lady", "Countess", "Sir", "Jonkheer", "Don", "Dona", "the Countess"}
_MILITARY_TITLES = {"Capt", "Col", "Major"}
_ACADEMIC_TITLES = {"Dr", "Rev"}

CAT_COLS = ["Sex", "Embarked", "Title", "AgeGroup", "FareBin", "Deck"]


def extract_title(name: str) -> str:
    title = name.split(",")[1].split(".")[0].strip()
    if title in _TITLE_MAP:
        return _TITLE_MAP[title]
    if title in _ROYAL_TITLES or title in _MILITARY_TITLES or title in _ACADEMIC_TITLES:
        return "Rare"
    return title


def extract_deck(cabin) -> str:
    if pd.isna(cabin) or cabin == "":
        return "Unknown"
    return str(cabin)[0]


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transformations that don't require fitting."""
    df = df.copy()

    df["Title"] = df["Name"].apply(extract_title)
    df["Deck"] = df["Cabin"].apply(extract_deck)
    df["TicketFreq"] = df["Ticket"].map(df["Ticket"].value_counts()).fillna(1).astype(int)

    age_medians = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_medians).fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["IsChild"] = (df["Age"] < 16).astype(int)
    df["IsMother"] = (
        (df["Sex"] == "female")
        & (df["Age"] > 18)
        & (df["Parch"] > 0)
        & (df["Title"] != "Miss")
    ).astype(int)
    df["AgePclass"] = df["Age"] * df["Pclass"]
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 120],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    ).astype(str)

    return df


def fit_transform(df: pd.DataFrame) -> tuple:
    """
    Fit encoders on training data and return (transformed_df, encoders).

    Parameters
    ----------
    df : raw training DataFrame (must contain 'Survived')

    Returns
    -------
    (X_df, encoders) where encoders is a dict to pass to transform()
    """
    df = _base_features(df)
    passenger_ids = df["PassengerId"].copy()

    # Fit FareBin quantile bins from training data
    _, fare_bins = pd.qcut(df["Fare"], q=4, labels=False, retbins=True)
    fare_bins[0] = -np.inf
    fare_bins[-1] = np.inf
    df["FareBin"] = pd.cut(
        df["Fare"], bins=fare_bins, labels=["Low", "Medium", "High", "VeryHigh"]
    ).astype(str)

    # Fit OrdinalEncoder on training data (handles unseen categories gracefully)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=float)
    df[CAT_COLS] = enc.fit_transform(df[CAT_COLS].astype(str))

    encoders = {"ordinal": enc, "fare_bins": fare_bins}

    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId", "SibSp", "Parch"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df, encoders


def transform(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Apply fitted encoders to test data.

    Parameters
    ----------
    df       : raw test DataFrame
    encoders : dict returned by fit_transform()

    Returns
    -------
    Transformed DataFrame with PassengerId retained.
    """
    df = _base_features(df)
    passenger_ids = df["PassengerId"].copy()

    fare_bins = encoders["fare_bins"]
    df["FareBin"] = pd.cut(
        df["Fare"], bins=fare_bins, labels=["Low", "Medium", "High", "VeryHigh"]
    ).astype(str)

    enc = encoders["ordinal"]
    df[CAT_COLS] = enc.transform(df[CAT_COLS].astype(str))

    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId", "SibSp", "Parch"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["PassengerId"] = passenger_ids.values
    return df
