from __future__ import annotations

from typing import Tuple

import pandas as pd

from config import DATA_PATH, GROUP_COLUMNS, TARGET_COLUMN


def load_clean_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Death Count and Predicted Value are strings with comma separators.
    df["death_count"] = pd.to_numeric(
        df["Death Count"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df["predicted_value"] = pd.to_numeric(
        df["Predicted Value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    df = df.dropna(subset=["Date", TARGET_COLUMN]).copy()
    df = df.sort_values(GROUP_COLUMNS + ["Date"]).reset_index(drop=True)
    return df


def build_supervised_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter

    grouped = df.groupby(GROUP_COLUMNS, observed=True)
    df["lag_1"] = grouped[TARGET_COLUMN].shift(1)
    df["lag_3"] = grouped[TARGET_COLUMN].shift(3)
    df["roll_mean_3"] = grouped[TARGET_COLUMN].shift(1).rolling(window=3).mean()
    df["roll_std_3"] = grouped[TARGET_COLUMN].shift(1).rolling(window=3).std()

    df = df.dropna(
        subset=["lag_1", "lag_3", "roll_mean_3", "roll_std_3", "predicted_value"]
    ).copy()
    return df


def get_feature_target_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        "State",
        "Indicator",
        "Percent Complete",
        "Percent Pending Investigation",
        "year",
        "month",
        "quarter",
        "lag_1",
        "lag_3",
        "roll_mean_3",
        "roll_std_3",
    ]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
