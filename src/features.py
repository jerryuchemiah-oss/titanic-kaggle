"""
Feature engineering pipeline for the Titanic Kaggle competition.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Title groups for better consolidation
_TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}

_ROYAL_TITLES = {"Lady", "Countess", "Sir", "Jonkheer", "Don", "Dona", "the Countess"}
_MILITARY_TITLES = {"Capt", "Col", "Major"}
_ACADEMIC_TITLES = {"Dr", "Rev"}


def extract_title(name: str) -> str:
    """Extract and normalize title from passenger name."""
    title = name.split(",")[1].split(".")[0].strip()

    if title in _TITLE_MAP:
        return _TITLE_MAP[title]
    if title in _ROYAL_TITLES or title in _MILITARY_TITLES or title in _ACADEMIC_TITLES:
        return "Rare"
    return title


def extract_deck(cabin) -> str:
    """Extract deck letter from Cabin value; return 'Unknown' if missing."""
    if pd.isna(cabin) or cabin == "":
        return "Unknown"
    return str(cabin)[0]


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

    # --- Deck: extract from Cabin before dropping ---
    df["Deck"] = df["Cabin"].apply(extract_deck)

    # --- TicketFreq: number of passengers sharing the same ticket ---
    ticket_counts = df["Ticket"].map(df["Ticket"].value_counts())
    df["TicketFreq"] = ticket_counts.fillna(1).astype(int)

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

    # --- IsChild: age < 16 ---
    df["IsChild"] = (df["Age"] < 16).astype(int)

    # --- IsMother: female, age > 18, has children (Parch > 0), not Miss ---
    df["IsMother"] = (
        (df["Sex"] == "female")
        & (df["Age"] > 18)
        & (df["Parch"] > 0)
        & (df["Title"] != "Miss")
    ).astype(int)

    # --- AgePclass: interaction feature ---
    df["AgePclass"] = df["Age"] * df["Pclass"]

    # --- FarePerPerson ---
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

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
    for col in ["Sex", "Embarked", "Title", "AgeGroup", "FareBin", "Deck"]:
        df[col] = le.fit_transform(df[col].astype(str))

    # --- Drop columns not useful for modelling ---
    # Drop Ticket and Cabin (already extracted), SibSp and Parch (FamilySize captures them)
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId", "SibSp", "Parch"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if not is_train:
        # Reattach PassengerId for later submission assembly
        df["PassengerId"] = passenger_ids.values

    return df
