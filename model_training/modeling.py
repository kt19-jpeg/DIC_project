from __future__ import annotations

from typing import Dict

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_STATE


def get_model_pipelines() -> Dict[str, Pipeline]:
    categorical_features = ["State", "Indicator"]
    numeric_features = [
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

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    scaled_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    plain_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    ridge_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", scaled_numeric_transformer, numeric_features),
        ]
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", plain_numeric_transformer, numeric_features),
        ]
    )

    return {
        "ridge_regression": Pipeline(
            steps=[
                ("preprocessor", ridge_preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        max_depth=8,
                        learning_rate=0.05,
                        n_estimators=450,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
