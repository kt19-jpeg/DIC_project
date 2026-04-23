from __future__ import annotations

import sys
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from artifacts import ensure_results_dir, write_json, write_pickle, write_text
from data_prep import build_supervised_features, get_feature_target_frames, load_clean_dataset
from evaluation import rank_models, regression_metrics
from modeling import get_model_pipelines
from splitter import temporal_train_val_test_split


def _prediction_frame(
    df: pd.DataFrame,
    mask: pd.Series,
    y_true: pd.Series,
    y_pred,
) -> List[Dict[str, object]]:
    indexed = df.loc[mask, ["Date", "State", "Indicator"]].copy()
    indexed["actual"] = y_true.values
    indexed["predicted"] = y_pred
    indexed["residual"] = indexed["actual"] - indexed["predicted"]
    indexed["Date"] = indexed["Date"].astype(str)
    return indexed.to_dict(orient="records")


def run_training_pipeline() -> None:
    output_dir = ensure_results_dir()
    print(f"Saving training results to: {output_dir}")

    df = load_clean_dataset()
    df_model = build_supervised_features(df)
    split = temporal_train_val_test_split(df_model)

    X, y = get_feature_target_frames(df_model)

    X_train = X.loc[split.train_mask]
    X_val = X.loc[split.val_mask]
    X_test = X.loc[split.test_mask]

    y_train = y.loc[split.train_mask]
    y_val = y.loc[split.val_mask]
    y_test = y.loc[split.test_mask]

    models = get_model_pipelines()

    metrics_rows: List[Dict[str, float]] = []
    predictions: Dict[str, Dict[str, List[Dict[str, object]]]] = {}

    model_items = list(models.items())
    progress = tqdm(
        model_items,
        desc="Training models",
        unit="model",
        file=sys.stdout,
        dynamic_ncols=True,
    )

    for model_name, pipeline in progress:
        progress.set_description(f"Training {model_name}")
        pipeline.fit(X_train, y_train)

        val_pred = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)

        val_scores = regression_metrics(y_val, val_pred)
        test_scores = regression_metrics(y_test, test_pred)

        metrics_rows.append(
            {
                "model": model_name,
                "val_mae": val_scores["mae"],
                "val_rmse": val_scores["rmse"],
                "val_r2": val_scores["r2"],
                "test_mae": test_scores["mae"],
                "test_rmse": test_scores["rmse"],
                "test_r2": test_scores["r2"],
            }
        )

        predictions[model_name] = {
            "validation": _prediction_frame(df_model, split.val_mask, y_val, val_pred),
            "test": _prediction_frame(df_model, split.test_mask, y_test, test_pred),
        }

        progress.set_postfix(val_rmse=f"{val_scores['rmse']:.3f}")
        write_pickle(f"{model_name}.pkl", pipeline)

    ranked_metrics = rank_models(metrics_rows)
    best_model_name = ranked_metrics[0]["model"]

    write_pickle("all_models.pkl", {name: pipe for name, pipe in models.items()})
    write_pickle("best_model.pkl", models[best_model_name])

    write_json("split_metadata.json", split.metadata)
    write_json("metrics_summary.json", ranked_metrics)
    write_json("predictions_by_model.json", predictions)
    write_text("best_model.txt", f"{best_model_name}\n")

    summary_lines = [
        "# Model Training Summary",
        "",
        f"- Training rows: {len(X_train)}",
        f"- Validation rows: {len(X_val)}",
        f"- Test rows: {len(X_test)}",
        f"- Best model (by validation RMSE): {best_model_name}",
        "",
        "## Metrics",
    ]
    for row in ranked_metrics:
        summary_lines.append(
            (
                f"- {row['model']}: "
                f"val_rmse={row['val_rmse']:.3f}, val_mae={row['val_mae']:.3f}, val_r2={row['val_r2']:.3f}, "
                f"test_rmse={row['test_rmse']:.3f}, test_mae={row['test_mae']:.3f}, test_r2={row['test_r2']:.3f}"
            )
        )
    write_text("training_summary.md", "\n".join(summary_lines) + "\n")

    print("Training complete.")
    print(f"Best model: {best_model_name}")
    for row in ranked_metrics:
        print(
            f"{row['model']}: "
            f"val_rmse={row['val_rmse']:.3f}, test_rmse={row['test_rmse']:.3f}"
        )


if __name__ == "__main__":
    run_training_pipeline()
