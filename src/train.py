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
    get_feature_minimization_groups,
    prepare_inference_features,
    split_features_target,
    time_train_test_split,
)
from .models import LogTargetModel, get_model_feature_importance, train_catboost, train_xgboost
from .utils import ensure_directories


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
        raise ValueError("Time split validation failed because match_date contains invalid values")
    if train_dates.max() > test_dates.min():
        raise ValueError("Time split validation failed: train period overlaps future test period")
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
        raise ValueError(f"Potential target leakage detected in features: {suspicious}")
    return {"suspicious_feature_count": 0}


def _fit_model_candidates(x_train, y_train, tune_xgb=False, tune_catboost=False):
    y_train_log = np.log1p(np.maximum(y_train.to_numpy(dtype=float), 0.0))

    xgb_model = LogTargetModel(base_model=train_xgboost(x_train=x_train, y_train=y_train_log, tune=tune_xgb))
    cat_model = LogTargetModel(base_model=train_catboost(x_train=x_train, y_train=y_train_log, tune=tune_catboost))

    return {
        "xgboost_log": xgb_model,
        "catboost_log": cat_model,
    }


def _score_candidates(models, x_train, y_train, x_test, y_test):
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

    preds = {}
    metrics_by_model = {}
    for name, values in baseline_preds.items():
        pred = np.maximum(np.asarray(values, dtype=float), 0.0)
        preds[name] = pred
        metrics_by_model[name] = compute_metrics(y_test, pred)

    for name, model in models.items():
        pred = np.maximum(np.asarray(model.predict(x_test), dtype=float), 0.0)
        preds[name] = pred
        metrics_by_model[name] = compute_metrics(y_test, pred)

    ranked = sorted(metrics_by_model.items(), key=lambda kv: (kv[1]["mae"], kv[1]["rmse"]))
    return preds, metrics_by_model, ranked


def _run_feature_minimization(match_level_df, x_train, y_train, x_test, y_test):
    groups = get_feature_minimization_groups(match_level_df)
    rows = []
    y_train_log = np.log1p(np.maximum(y_train.to_numpy(dtype=float), 0.0))

    for name, columns in groups.items():
        xtr = x_train[columns].copy()
        xte = x_test[columns].copy()
        model = LogTargetModel(base_model=train_xgboost(x_train=xtr, y_train=y_train_log, tune=False))
        pred = np.maximum(np.asarray(model.predict(xte), dtype=float), 0.0)
        metrics = compute_metrics(y_test, pred)
        rows.append({
            "feature_set": name,
            "feature_count": int(len(columns)),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "mape": float(metrics["mape"]),
            "r2": float(metrics["r2"]),
        })

    results_df = pd.DataFrame(rows).sort_values(["mae", "feature_count"]).reset_index(drop=True)
    save_csv(results_df, OUTPUTS_DIR / "minimal_feature_results.csv")
    if len(results_df) > 0:
        best = results_df.iloc[0]
        text = [
            "Minimal Feature Set Evaluation",
            f"Best set: {best['feature_set']}",
            f"Features: {int(best['feature_count'])}",
            f"MAE: {float(best['mae']):.2f}",
            f"RMSE: {float(best['rmse']):.2f}",
        ]
        (OUTPUTS_DIR / "minimal_feature_summary.txt").write_text("\n".join(text), encoding="utf-8")
    return results_df


def run_pipeline(data_dir=DEFAULT_DATA_DIR, tune_xgb=False, tune_catboost=False):
    plots_dir = OUTPUTS_DIR / "plots"
    ensure_directories([OUTPUTS_DIR, PREDICTIONS_DIR, FEATURE_IMPORTANCE_DIR, MODELS_DIR, REPORTS_DIR, plots_dir])

    validate_required_data_files(data_dir)
    tables = load_raw_tables(data_dir)
    match_level_df, dataset_stats = build_match_level_dataset(tables, use_weather_api=False, return_stats=True)

    feature_columns = get_feature_columns(match_level_df)
    x, y, meta = split_features_target(match_level_df, feature_columns)

    if len(x) < 8:
        raise ValueError("Not enough matches to create robust train and test sets. At least 8 rows are required.")

    leakage_stats = _validate_leakage(x=x, y=y)

    x_train, x_test, y_train, y_test, meta_train, meta_test = time_train_test_split(
        x=x,
        y=y,
        meta=meta,
        test_size=TEST_SIZE,
    )
    split_stats = _validate_time_split(meta_train=meta_train, meta_test=meta_test)

    candidate_models = _fit_model_candidates(
        x_train=x_train,
        y_train=y_train,
        tune_xgb=tune_xgb,
        tune_catboost=tune_catboost,
    )

    predictions_dict, metrics_by_model, ranked = _score_candidates(
        models=candidate_models,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    best_model_name = ranked[0][0]
    best_metrics = ranked[0][1]
    best_model = candidate_models.get(best_model_name)
    if best_model is None:
        best_model_name = "xgboost_log"
        best_model = candidate_models[best_model_name]
        best_metrics = metrics_by_model[best_model_name]

    comparison_df = build_comparison_table(
        {k: {"metrics": v} for k, v in metrics_by_model.items()}
    )
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

    minimal_feature_results = _run_feature_minimization(match_level_df, x_train, y_train, x_test, y_test)

    run_info = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "best_model_name": best_model_name,
        "best_model_base_name": "xgboost" if "xgboost" in best_model_name else "catboost" if "catboost" in best_model_name else "baseline",
        "model_type": "log",
        "log_transform_used": True,
        "weighted_training_used": False,
        "primary_candidate": PRIMARY_MODEL_CANDIDATE,
        "tested_model_names": list(metrics_by_model.keys()),
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
        "model_ranking": [{"model": name, "mae": float(payload["mae"]), "rmse": float(payload["rmse"])} for name, payload in ranked],
        "time_split": split_stats,
        "leakage_validation": leakage_stats,
        "data_dir": str(Path(data_dir)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_enabled": False,
        "calibration_type": None,
        "calibration_params": None,
        "residual_correction_enabled": False,
        "minimal_feature_best_set": minimal_feature_results.iloc[0].to_dict() if len(minimal_feature_results) > 0 else {},
        "tuning": {"xgboost": bool(tune_xgb), "catboost": bool(tune_catboost)},
    }

    with open(BEST_MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    summary_lines = [
        "Attendance Prediction Project Report",
        f"Best model: {best_model_name}",
        f"MAE: {best_metrics['mae']:.2f}",
        f"RMSE: {best_metrics['rmse']:.2f}",
        f"R2: {best_metrics['r2']:.4f}",
        f"MAPE: {best_metrics['mape']:.2f}",
        f"Median Abs Error: {best_metrics['median_abs_error']:.2f}",
        f"Rows train: {len(x_train)}",
        f"Rows test: {len(x_test)}",
        f"Split train max date: {split_stats['train_max_date']}",
        f"Split test min date: {split_stats['test_min_date']}",
        f"Transfermarkt opponents with priors: {int(dataset_stats.get('transfermarkt', {}).get('opponents_with_priors', 0))}",
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
    parser.add_argument("--tune-catboost", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.check_data_only:
        validate_required_data_files(data_dir)
        print(f"Data check passed: {data_dir}")
        return

    result = run_pipeline(data_dir=data_dir, tune_xgb=args.tune_xgb, tune_catboost=args.tune_catboost)

    metrics = dict(result["run_info"]["best_metrics"])
    print(f"Best model: {result['run_info']['best_model_name']}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}")
    print(f"R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()

