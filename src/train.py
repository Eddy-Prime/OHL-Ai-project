from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np

from .config import (
    BEST_MODEL_ARTIFACT_PATH,
    BEST_MODEL_METADATA_PATH,
    DEFAULT_DATA_DIR,
    FEATURE_IMPORTANCE_DIR,
    MODELS_DIR,
    MODEL_SCHEMA_VERSION,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    PRIMARY_MODEL_CANDIDATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data_loader import load_raw_tables, validate_required_data_files
from .evaluate import (
    build_predictions_table,
    compute_metrics,
    plot_actual_vs_predicted,
    plot_attendance_distribution,
    plot_attendance_over_time,
    plot_feature_importance,
    plot_residual_distribution,
    save_csv,
    save_feature_importance,
)
from .features import (
    build_feature_fill_values,
    build_match_level_dataset,
    get_feature_columns,
    split_features_target,
    time_train_test_split,
)
from .models import LogTargetModel, get_model_feature_importance, train_xgboost
from .utils import ensure_directories

MODEL_NAME = "xgboost_log"


def _build_top_error_cases(predictions_df, model_name):
    error_column = f"abs_error_{model_name}"
    pred_column = f"pred_{model_name}"

    top_errors = predictions_df.sort_values(error_column, ascending=False).head(10).copy()
    top_errors["signed_error"] = top_errors["actual"] - top_errors[pred_column]
    top_errors["pct_error"] = np.where(
        top_errors["actual"] != 0,
        np.abs(top_errors["signed_error"]) / np.abs(top_errors["actual"]) * 100.0,
        np.nan,
    )
    return top_errors


def run_pipeline(data_dir=DEFAULT_DATA_DIR, tune_xgb=False):
    ensure_directories([OUTPUTS_DIR, PREDICTIONS_DIR, FEATURE_IMPORTANCE_DIR, MODELS_DIR, REPORTS_DIR])

    validate_required_data_files(data_dir)
    tables = load_raw_tables(data_dir)
    match_level_df, dataset_stats = build_match_level_dataset(tables, use_weather_api=False, return_stats=True)

    feature_columns = get_feature_columns(match_level_df)
    x, y, meta = split_features_target(match_level_df, feature_columns)

    if len(x) < 2:
        raise ValueError("Not enough matches to create train and test sets. At least 2 rows are required.")

    x_train, x_test, y_train, y_test, _, meta_test = time_train_test_split(
        x=x,
        y=y,
        meta=meta,
        test_size=TEST_SIZE,
    )

    y_train_log = np.log1p(np.maximum(y_train.to_numpy(dtype=float), 0.0))
    xgb_model = train_xgboost(x_train=x_train, y_train=y_train_log, tune=tune_xgb)
    model = LogTargetModel(base_model=xgb_model)

    predictions = np.asarray(model.predict(x_test), dtype=float)
    predictions = np.maximum(predictions, 0.0)
    metrics = compute_metrics(y_test, predictions)

    predictions_df = build_predictions_table(
        meta_test=meta_test,
        y_test=y_test,
        predictions_dict={MODEL_NAME: predictions},
    )
    save_csv(predictions_df, PREDICTIONS_DIR / "test_predictions.csv")

    top_errors = _build_top_error_cases(predictions_df=predictions_df, model_name=MODEL_NAME)
    save_csv(top_errors, PREDICTIONS_DIR / "top_error_cases.csv")

    feature_names, feature_values = get_model_feature_importance(model)
    feature_importance_df = save_feature_importance(
        names=feature_names,
        values=feature_values,
        output_path=FEATURE_IMPORTANCE_DIR / "best_model_feature_importance.csv",
        top_n=30,
    )

    joblib.dump(model, BEST_MODEL_ARTIFACT_PATH)

    plot_actual_vs_predicted(
        y_true=y_test.to_numpy(dtype=float),
        y_pred=predictions,
        output_path=PREDICTIONS_DIR / "actual_vs_predicted_best_model.png",
    )
    plot_feature_importance(
        feature_importance_df=feature_importance_df,
        output_path=FEATURE_IMPORTANCE_DIR / "top_feature_importance.png",
    )
    plot_attendance_over_time(
        meta_test=meta_test,
        y_true=y_test,
        y_pred=predictions,
        output_path=PREDICTIONS_DIR / "attendance_over_time_test.png",
    )
    plot_attendance_distribution(
        y_values=match_level_df[TARGET_COLUMN],
        output_path=PREDICTIONS_DIR / "attendance_distribution.png",
    )
    plot_residual_distribution(
        y_true=y_test,
        y_pred=predictions,
        output_path=PREDICTIONS_DIR / "residual_distribution_best_model.png",
    )

    feature_fill_values = build_feature_fill_values(x_train=x_train, feature_columns=feature_columns)

    run_info = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "best_model_name": MODEL_NAME,
        "best_model_base_name": "xgboost",
        "model_type": "log",
        "log_transform_used": True,
        "weighted_training_used": False,
        "primary_candidate": PRIMARY_MODEL_CANDIDATE,
        "tested_model_names": [MODEL_NAME],
        "rows_total": int(len(match_level_df)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "evaluation_dataset": "OH Leuven internal only",
        "features_used": feature_columns,
        "user_input_features": ["match_date", "away_team", "stage", "kickoff_time"],
        "feature_fill_values": feature_fill_values,
        "use_weather_api": False,
        "weather_enrichment_stats": dataset_stats.get("weather", {}),
        "target_column": TARGET_COLUMN,
        "metrics_by_model": {MODEL_NAME: metrics},
        "best_metrics": metrics,
        "data_dir": str(Path(data_dir)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_enabled": False,
        "calibration_type": None,
        "calibration_params": None,
        "residual_correction_enabled": False,
    }

    with open(BEST_MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    summary_lines = [
        "Attendance Prediction Project Report",
        f"Best model: {MODEL_NAME}",
        f"MAE: {metrics['mae']:.2f}",
        f"RMSE: {metrics['rmse']:.2f}",
        f"R2: {metrics['r2']:.4f}",
        f"MAPE: {metrics['mape']:.2f}",
        f"Median Abs Error: {metrics['median_abs_error']:.2f}",
        f"Rows train: {len(x_train)}",
        f"Rows test: {len(x_test)}",
        f"Data dir: {Path(data_dir)}",
    ]
    (REPORTS_DIR / "summary_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "dataset": match_level_df,
        "predictions": predictions_df,
        "top_errors": top_errors,
        "feature_importance": feature_importance_df,
        "run_info": run_info,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--tune-xgb", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.check_data_only:
        validate_required_data_files(data_dir)
        print(f"Data check passed: {data_dir}")
        return

    result = run_pipeline(data_dir=data_dir, tune_xgb=args.tune_xgb)

    metrics = dict(result["run_info"]["best_metrics"])
    print(f"Best model: {result['run_info']['best_model_name']}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}")
    print(f"R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()

