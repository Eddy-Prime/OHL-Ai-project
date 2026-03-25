from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .baselines import mean_baseline, opponent_mean_baseline
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
    build_comparison_table,
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
    prepare_inference_features,
    split_features_target,
    time_train_test_split,
)
from .models import LogTargetModel, get_model_feature_importance, train_catboost, train_xgboost
from .utils import ensure_directories
from .validation import time_series_rolling_split, backtest_model, RollingBacktestReport


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


def _validate_time_split(meta_train, meta_test):
    train_dates = pd.to_datetime(meta_train["match_date"], errors="coerce")
    test_dates = pd.to_datetime(meta_test["match_date"], errors="coerce")
    if train_dates.isna().any() or test_dates.isna().any():
        raise ValueError("Time split validation failed: invalid dates")
    if train_dates.max() >= test_dates.min():
        raise ValueError("Time split validation failed: train period overlaps test")
    return {
        "train_min_date": train_dates.min().date().isoformat(),
        "train_max_date": train_dates.max().date().isoformat(),
        "test_min_date": test_dates.min().date().isoformat(),
        "test_max_date": test_dates.max().date().isoformat(),
    }


def _validate_leakage(x, y):
    suspicious = []
    target = np.asarray(y, dtype=float)
    for col in x.columns:
        series = pd.to_numeric(x[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        aligned = series.to_numpy(dtype=float)
        mask = np.isfinite(aligned) & np.isfinite(target)
        if mask.sum() < max(5, int(0.2 * len(target))):
            continue
        equality_ratio = float(np.mean(np.isclose(aligned[mask], target[mask], atol=1e-8)))
        if equality_ratio > 0.95:
            suspicious.append({"feature": col, "equality_ratio": equality_ratio})
    if len(suspicious) > 0:
        raise ValueError(f"Potential target leakage detected: {suspicious}")
    return {"suspicious_feature_count": 0}


def run_research_pipeline(data_dir=DEFAULT_DATA_DIR, tune_xgb=False, tune_catboost=True, n_rolling_folds=5):
    plots_dir = OUTPUTS_DIR / "plots"
    ensure_directories([OUTPUTS_DIR, PREDICTIONS_DIR, FEATURE_IMPORTANCE_DIR, MODELS_DIR, REPORTS_DIR, plots_dir])

    validate_required_data_files(data_dir)
    tables = load_raw_tables(data_dir)
    match_level_df, dataset_stats = build_match_level_dataset(tables, use_weather_api=False, return_stats=True)

    feature_columns = get_feature_columns(match_level_df)
    x, y, meta = split_features_target(match_level_df, feature_columns)

    if len(x) < 16:
        raise ValueError("Not enough matches for rolling backtesting. Need at least 16 rows.")

    leakage_stats = _validate_leakage(x=x, y=y)

    x_train, x_test, y_train, y_test, meta_train, meta_test = time_train_test_split(
        x=x,
        y=y,
        meta=meta,
        test_size=TEST_SIZE,
    )
    split_stats = _validate_time_split(meta_train=meta_train, meta_test=meta_test)

    backtest_report = RollingBacktestReport()

    y_train_log = np.log1p(np.maximum(y_train.to_numpy(dtype=float), 0.0))

    def xgb_wrapper(x_tr, y_tr):
        return LogTargetModel(base_model=train_xgboost(x_train=x_tr, y_train=np.log1p(np.maximum(y_tr.to_numpy(dtype=float), 0.0)), tune=tune_xgb))

    def cat_wrapper(x_tr, y_tr):
        return LogTargetModel(base_model=train_catboost(x_train=x_tr, y_train=np.log1p(np.maximum(y_tr.to_numpy(dtype=float), 0.0)), tune=tune_catboost))

    rolling_folds = time_series_rolling_split(x_train, y_train, meta_train, n_folds=n_rolling_folds, test_size_ratio=0.2)

    xgb_backtest_df, xgb_summary = backtest_model(xgb_wrapper, x_train, y_train, meta_train, rolling_folds, "xgboost_log")
    if len(xgb_backtest_df) > 0:
        backtest_report.add_backtest("xgboost_log", xgb_backtest_df, xgb_summary)

    cat_backtest_df, cat_summary = backtest_model(cat_wrapper, x_train, y_train, meta_train, rolling_folds, "catboost_log")
    if len(cat_backtest_df) > 0:
        backtest_report.add_backtest("catboost_log", cat_backtest_df, cat_summary)

    save_csv(xgb_backtest_df, OUTPUTS_DIR / "backtest_xgboost.csv")
    save_csv(cat_backtest_df, OUTPUTS_DIR / "backtest_catboost.csv")

    xgb_model = LogTargetModel(base_model=train_xgboost(x_train=x_train, y_train=y_train_log, tune=tune_xgb))
    cat_model = LogTargetModel(base_model=train_catboost(x_train=x_train, y_train=y_train_log, tune=tune_catboost))

    fallback_mean = float(np.mean(y_train))
    baseline_preds = {
        "mean_baseline": mean_baseline(y_train=y_train, size=len(x_test)),
        "opponent_mean_baseline": opponent_mean_baseline(
            y_train=y_train,
            x_train=x_train,
            x_test=x_test,
            fallback_value=fallback_mean,
        ),
    }

    predictions_dict = {}
    metrics_by_model = {}
    for name, values in baseline_preds.items():
        pred = np.maximum(np.asarray(values, dtype=float), 0.0)
        predictions_dict[name] = pred
        metrics_by_model[name] = compute_metrics(y_test, pred)

    xgb_test_preds = np.maximum(np.asarray(xgb_model.predict(x_test), dtype=float), 0.0)
    predictions_dict["xgboost_log"] = xgb_test_preds
    metrics_by_model["xgboost_log"] = compute_metrics(y_test, xgb_test_preds)
    backtest_report.add_final_holdout("xgboost_log", metrics_by_model["xgboost_log"])

    cat_test_preds = np.maximum(np.asarray(cat_model.predict(x_test), dtype=float), 0.0)
    predictions_dict["catboost_log"] = cat_test_preds
    metrics_by_model["catboost_log"] = compute_metrics(y_test, cat_test_preds)
    backtest_report.add_final_holdout("catboost_log", metrics_by_model["catboost_log"])

    ranked = sorted(metrics_by_model.items(), key=lambda kv: (kv[1]["mae"], kv[1]["rmse"]))
    best_model_name = ranked[0][0]
    best_metrics = ranked[0][1]
    best_model = cat_model if best_model_name == "catboost_log" else xgb_model if best_model_name == "xgboost_log" else None

    if best_model is None:
        best_model_name = "catboost_log" if cat_summary.get("mae_mean", float("inf")) <= xgb_summary.get("mae_mean", float("inf")) else "xgboost_log"
        best_model = cat_model if best_model_name == "catboost_log" else xgb_model
        best_metrics = metrics_by_model[best_model_name]

    comparison_df = build_comparison_table({k: {"metrics": v} for k, v in metrics_by_model.items()})
    save_csv(comparison_df, OUTPUTS_DIR / "model_comparison.csv")

    predictions_df = build_predictions_table(
        meta_test=meta_test,
        y_test=y_test,
        predictions_dict=predictions_dict,
    )
    save_csv(predictions_df, PREDICTIONS_DIR / "test_predictions.csv")

    top_errors = _build_top_error_cases(predictions_df=predictions_df, model_name=best_model_name)
    save_csv(top_errors, PREDICTIONS_DIR / "top_error_cases.csv")

    feature_names, feature_values = get_model_feature_importance(best_model)
    feature_importance_df = save_feature_importance(
        names=feature_names,
        values=feature_values,
        output_path=FEATURE_IMPORTANCE_DIR / "best_model_feature_importance.csv",
        top_n=40,
    )

    joblib.dump(best_model, BEST_MODEL_ARTIFACT_PATH)

    best_predictions = np.maximum(np.asarray(predictions_dict[best_model_name], dtype=float), 0.0)
    plot_actual_vs_predicted(
        y_true=y_test.to_numpy(dtype=float),
        y_pred=best_predictions,
        output_path=PREDICTIONS_DIR / "actual_vs_predicted_best_model.png",
    )
    plot_feature_importance(
        feature_importance_df=feature_importance_df,
        output_path=FEATURE_IMPORTANCE_DIR / "top_feature_importance.png",
    )
    plot_attendance_over_time(
        meta_test=meta_test,
        y_true=y_test,
        y_pred=best_predictions,
        output_path=PREDICTIONS_DIR / "attendance_over_time_test.png",
    )
    plot_attendance_distribution(
        y_values=match_level_df[TARGET_COLUMN],
        output_path=PREDICTIONS_DIR / "attendance_distribution.png",
    )
    plot_residual_distribution(
        y_true=y_test,
        y_pred=best_predictions,
        output_path=PREDICTIONS_DIR / "residual_distribution_best_model.png",
    )

    plot_actual_vs_predicted(
        y_true=y_test.to_numpy(dtype=float),
        y_pred=best_predictions,
        output_path=plots_dir / "actual_vs_predicted.png",
    )
    plot_residual_distribution(
        y_true=y_test,
        y_pred=best_predictions,
        output_path=plots_dir / "residuals.png",
    )

    feature_fill_values = build_feature_fill_values(x_train=x_train, feature_columns=feature_columns)
    _ = prepare_inference_features(match_level_df.tail(1), feature_columns, fill_values=feature_fill_values)

    backtest_report.save_to_json(OUTPUTS_DIR / "rolling_backtest_summary.json")
    backtest_report.save_fold_details(OUTPUTS_DIR / "rolling_backtest_details.csv")

    run_info = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "best_model_name": best_model_name,
        "best_model_base_name": "catboost" if "catboost" in best_model_name else "xgboost",
        "model_type": "log",
        "log_transform_used": True,
        "rows_total": int(len(match_level_df)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "evaluation_dataset": "OH Leuven internal only",
        "features_used": feature_columns,
        "user_input_features": ["match_date", "away_team", "stage", "kickoff_time"],
        "feature_fill_values": feature_fill_values,
        "use_weather_api": False,
        "weather_enrichment_stats": dataset_stats.get("weather", {}),
        "transfermarkt_usage": dataset_stats.get("transfermarkt", {}),
        "target_column": TARGET_COLUMN,
        "metrics_by_model": metrics_by_model,
        "best_metrics": best_metrics,
        "holdout_test_metrics": best_metrics,
        "rolling_backtest_results": backtest_report.to_dict(),
        "time_split": split_stats,
        "leakage_validation": leakage_stats,
        "data_dir": str(Path(data_dir)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(BEST_MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    summary_lines = [
        "Attendance Prediction - Research Pipeline Report",
        f"Best model: {best_model_name}",
        f"Final holdout MAE: {best_metrics['mae']:.2f}",
        f"Final holdout RMSE: {best_metrics['rmse']:.2f}",
        f"Final holdout R2: {best_metrics['r2']:.4f}",
        f"Final holdout MAPE: {best_metrics['mape']:.2f}",
        f"Rows train: {len(x_train)}",
        f"Rows test: {len(x_test)}",
        f"Rolling backtest folds: {len(rolling_folds)}",
        f"Backtest MAE mean: {backtest_report.backtest_results.get(best_model_name, {}).get('summary', {}).get('mae_mean', float('nan')):.2f}",
        f"Data dir: {Path(data_dir)}",
    ]
    (REPORTS_DIR / "summary_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "dataset": match_level_df,
        "predictions": predictions_df,
        "top_errors": top_errors,
        "feature_importance": feature_importance_df,
        "run_info": run_info,
        "backtest_report": backtest_report,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--tune-xgb", action="store_true")
    parser.add_argument("--tune-catboost", action="store_true")
    parser.add_argument("--n-rolling-folds", type=int, default=5)
    parser.add_argument("--check-data-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.check_data_only:
        validate_required_data_files(data_dir)
        print(f"Data check passed: {data_dir}")
        return

    result = run_research_pipeline(data_dir=data_dir, tune_xgb=args.tune_xgb, tune_catboost=args.tune_catboost, n_rolling_folds=args.n_rolling_folds)

    metrics = dict(result["run_info"]["best_metrics"])
    print(f"Best model: {result['run_info']['best_model_name']}")
    print(f"Final holdout MAE: {metrics['mae']:.2f}")
    print(f"Final holdout RMSE: {metrics['rmse']:.2f}")
    print(f"Final holdout R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()

