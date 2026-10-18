"""Data validation and target normalization for the UCI Heart Disease dataset."""
from pathlib import Path

import pandas as pd

TARGET_CANDIDATES = ("target", "num", "diagnosis", "heart_disease")


def load_and_clean(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.replace({"?": pd.NA, "": pd.NA, "NA": pd.NA, "nan": pd.NA})
    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"Target column not found. Expected one of {TARGET_CANDIDATES}")
    df = df.dropna(subset=[target_col]).copy()
    # UCI Cleveland target: 0 = no disease; 1..4 = presence of disease.
    target = pd.to_numeric(df[target_col], errors="coerce")
    valid = target.notna()
    df = df.loc[valid].copy()
    df["target"] = (target.loc[valid].astype(float) > 0).astype(int)
    if df["target"].nunique() != 2:
        raise ValueError("Target must contain both binary classes after normalization")
    df = df.drop(columns=[c for c in TARGET_CANDIDATES if c != "target" and c in df.columns])
    for col in df.columns:
        if col != "target":
            # Convert numeric-looking strings; leave truly categorical values unchanged.
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    if "target" not in df:
        raise ValueError("Cleaned dataframe must contain 'target'")
    X = df.drop(columns="target")
    y = df["target"].astype(int)
    if X.empty:
        raise ValueError("No feature columns found")
    return X, y
