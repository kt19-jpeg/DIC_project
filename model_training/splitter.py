from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from config import DATE_COLUMN, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


@dataclass
class SplitResult:
    train_mask: pd.Series
    val_mask: pd.Series
    test_mask: pd.Series
    metadata: Dict[str, object]


def temporal_train_val_test_split(df: pd.DataFrame) -> SplitResult:
    if round(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 2) != 1.00:
        raise ValueError("Split ratios must add up to 1.0")

    unique_dates = sorted(df[DATE_COLUMN].dropna().unique())
    total_dates = len(unique_dates)
    if total_dates < 10:
        raise ValueError("Not enough time points to create train/val/test split")

    train_end_idx = int(total_dates * TRAIN_RATIO)
    val_end_idx = int(total_dates * (TRAIN_RATIO + VAL_RATIO))

    # Keep at least one date in each split.
    train_end_idx = max(1, min(train_end_idx, total_dates - 2))
    val_end_idx = max(train_end_idx + 1, min(val_end_idx, total_dates - 1))

    train_dates = set(unique_dates[:train_end_idx])
    val_dates = set(unique_dates[train_end_idx:val_end_idx])
    test_dates = set(unique_dates[val_end_idx:])

    train_mask = df[DATE_COLUMN].isin(train_dates)
    val_mask = df[DATE_COLUMN].isin(val_dates)
    test_mask = df[DATE_COLUMN].isin(test_dates)

    metadata = {
        "total_rows": int(len(df)),
        "total_dates": total_dates,
        "date_start": str(min(unique_dates)),
        "date_end": str(max(unique_dates)),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_start": str(min(train_dates)),
        "train_end": str(max(train_dates)),
        "val_start": str(min(val_dates)),
        "val_end": str(max(val_dates)),
        "test_start": str(min(test_dates)),
        "test_end": str(max(test_dates)),
    }
    return SplitResult(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, metadata=metadata)
