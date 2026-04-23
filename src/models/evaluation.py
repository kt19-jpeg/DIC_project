from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def rank_models(metrics_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    return sorted(
        metrics_rows,
        key=lambda row: (
            row["val_rmse"],
            row["val_mae"],
        ),
    )
