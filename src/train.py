from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import joblib

from .baselines import mean_baseline, opponent_mean_baseline
from .calibration import (
    fit_multiplicative_calibrator,
    apply_multiplicative_calibrator,
    fit_linear_calibrator,
    apply_linear_calibrator,
    linear_calibrator_params,
)
from .config import (
    BEST_MODEL_ARTIFACT_PATH,
    BEST_MODEL_METADATA_PATH,
    DEFAULT_DATA_DIR,
    FEATURE_IMPORTANCE_DIR,
    MODEL_SCHEMA_VERSION,
    MODELS_DIR,
    OUTPUTS_DIR,
    PRIMARY_MODEL_CANDIDATE,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    EXTERNAL_TRAINING_SUMMARY_PATH,
    TRANSFERMARKT_EXTERNAL_PATH,
    WEATHER_API_ENABLED_DEFAULT,
    WEATHER_IMPACT_COMPARISON_PATH,
)
from .data_loader import load_raw_tables
from .external_data import load_external_transfermarkt_training_rows
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
    get_full_reference_feature_columns,
    get_feature_minimization_groups,
    split_features_target,
    time_train_test_split,
)
from .models import get_model_feature_importance, train_linear_regression, train_random_forest, train_xgboost
from .models import LogTargetModel, WeightedBlendModel
from .utils import ensure_directories

MODEL_CANDIDATES = ["linear_regression", "random_forest", "xgboost"]
BEST_SELECTION_CANDIDATES = [
    "linear_regression_raw",
    "linear_regression_calibrated",
    "random_forest_raw",
    "random_forest_calibrated",
    "xgboost_raw",
    "xgboost_log",
    "xgboost_log_weighted",
    "xgboost_calibrated",
    "weighted_blend_raw",
]
ENSEMBLE_WEIGHTS = {"xgboost": 0.7, "random_forest": 0.3}
RANDOM_FOREST_ARTIFACT_PATH = MODELS_DIR / "random_forest_model.joblib"
XGBOOST_ARTIFACT_PATH = MODELS_DIR / "xgboost_model.joblib"
MINIMAL_RESULTS_PATH = OUTPUTS_DIR / "minimal_feature_results.csv"
MINIMAL_SUMMARY_PATH = OUTPUTS_DIR / "minimal_feature_summary.txt"
MINIMAL_PLOT_PATH = OUTPUTS_DIR / "minimal_feature_plot.png"
CALIBRATION_VALIDATION_SIZE = 0.2
WEIGHT_ALPHA_DEFAULT = 1.0


def run_weather_impact_comparison(
    data_dir=DEFAULT_DATA_DIR,
    tune_rf=False,
    tune_xgb=False,
    tune_catboost=False,
    use_log_target=False,
    run_ablation=False,
    run_feature_minimization=False,
    use_calibration=False,
):
    without_weather = run_pipeline(
        data_dir=data_dir,
        tune_rf=tune_rf,
        tune_xgb=tune_xgb,
        tune_catboost=tune_catboost,
        use_log_target=use_log_target,
        run_ablation=run_ablation,
        run_feature_minimization=run_feature_minimization,
        use_calibration=use_calibration,
        use_weather_api=False,
    )
    with_weather = run_pipeline(
        data_dir=data_dir,
        tune_rf=tune_rf,
        tune_xgb=tune_xgb,
        tune_catboost=tune_catboost,
        use_log_target=use_log_target,
        run_ablation=run_ablation,
        run_feature_minimization=run_feature_minimization,
        use_calibration=use_calibration,
        use_weather_api=True,
    )

    without_df = without_weather["comparison"][["model", "mae", "rmse", "mape"]].rename(
        columns={"mae": "mae_without_weather", "rmse": "rmse_without_weather", "mape": "mape_without_weather"}
    )
    with_df = with_weather["comparison"][["model", "mae", "rmse", "mape"]].rename(
        columns={"mae": "mae_with_weather", "rmse": "rmse_with_weather", "mape": "mape_with_weather"}
    )
    comparison = without_df.merge(with_df, on="model", how="outer")
    comparison["delta_mae_pct"] = np.where(
        comparison["mae_without_weather"].notna() & (comparison["mae_without_weather"] != 0),
        (comparison["mae_with_weather"] - comparison["mae_without_weather"]) / comparison["mae_without_weather"] * 100.0,
        np.nan,
    )
    comparison = comparison.sort_values("model").reset_index(drop=True)
    save_csv(comparison, WEATHER_IMPACT_COMPARISON_PATH)
    return {
        "without_weather": without_weather,
        "with_weather": with_weather,
        "comparison": comparison,
        "output_path": str(WEATHER_IMPACT_COMPARISON_PATH),
    }


def _fit_model(model_name, x_train, y_train, tune_rf, tune_xgb, sample_weight=None):
    if model_name == "linear_regression":
        return train_linear_regression(x_train=x_train, y_train=y_train, sample_weight=sample_weight)
    if model_name == "random_forest":
        return train_random_forest(x_train=x_train, y_train=y_train, tune=tune_rf, sample_weight=sample_weight)
    if model_name == "xgboost":
        return train_xgboost(x_train=x_train, y_train=y_train, tune=tune_xgb, sample_weight=sample_weight)
    raise ValueError(f"Unknown model candidate: {model_name}")


def _inverse_mae_weights(mae_by_model):
    safe = {}
    for name, value in mae_by_model.items():
        safe[name] = max(float(value), 1e-9)
    inv = {name: 1.0 / value for name, value in safe.items()}
    total = float(sum(inv.values()))
    return {name: float(inv[name] / total) for name in inv}


def _calibration_payload_from_validation(y_true, y_pred):
    linear_payload = _fit_default_calibrator(y_true=y_true, y_pred=y_pred)
    linear_pred = _apply_calibrator(
        y_pred=y_pred,
        calibration_type=linear_payload["calibration_type"],
        calibration_object=linear_payload["calibration_object"],
    )
    linear_mae = compute_metrics(y_true, linear_pred)["mae"]

    mult_k = fit_multiplicative_calibrator(y_true=y_true, y_pred=y_pred)
    mult_pred = apply_multiplicative_calibrator(y_pred=y_pred, k=mult_k)
    mult_mae = compute_metrics(y_true, mult_pred)["mae"]

    if mult_mae < linear_mae:
        return {
            "enabled": True,
            "type": "multiplicative",
            "params": {"k": float(mult_k)},
            "mae": float(mult_mae),
        }
    return {
        "enabled": True,
        "type": "linear",
        "params": linear_payload["calibration_params"],
        "mae": float(linear_mae),
    }


def _build_weighted_blend_payload(x_train, y_train, x_test, y_test, tune_rf, tune_xgb, use_calibration):
    x_base, y_base, x_val, y_val = _split_train_for_calibration(x_train, y_train)
    component_names = ["xgboost", "random_forest", "linear_regression"]

    validation_mae = {}
    calibration_cfg = {}

    for name in component_names:
        base_model = _fit_model(name, x_base, y_base, tune_rf=tune_rf, tune_xgb=tune_xgb)
        val_pred_raw = _predict_non_negative(base_model, x_val)
        raw_mae = float(compute_metrics(y_val, val_pred_raw)["mae"])

        if use_calibration and len(x_val) >= 4:
            cal_payload = _calibration_payload_from_validation(y_true=y_val, y_pred=val_pred_raw)
            cal_mae = float(cal_payload["mae"])
            if cal_mae < raw_mae:
                validation_mae[name] = cal_mae
                calibration_cfg[name] = {
                    "enabled": True,
                    "type": cal_payload["type"],
                    "params": cal_payload["params"],
                }
            else:
                validation_mae[name] = raw_mae
                calibration_cfg[name] = {"enabled": False, "type": None, "params": None}
        else:
            validation_mae[name] = raw_mae
            calibration_cfg[name] = {"enabled": False, "type": None, "params": None}

    blend_weights = _inverse_mae_weights(validation_mae)

    trained_models = {
        name: _fit_model(name, x_train, y_train, tune_rf=tune_rf, tune_xgb=tune_xgb)
        for name in component_names
    }
    blend_model = WeightedBlendModel(models_dict=trained_models, weights_dict=blend_weights, calibration_dict=calibration_cfg)
    blend_pred = _predict_non_negative(blend_model, x_test)
    blend_metrics = compute_metrics(y_test, blend_pred)

    return {
        "weighted_blend_raw": {
            "model": blend_model,
            "model_base_name": "weighted_blend",
            "metrics": blend_metrics,
            "predictions": blend_pred,
            "calibration_enabled": any(bool(v.get("enabled", False)) for v in calibration_cfg.values()),
            "calibration_type": "component",
            "calibration_params": calibration_cfg,
            "raw_metrics": blend_metrics,
            "calibrated_metrics": blend_metrics,
            "component_weights": blend_weights,
            "component_validation_mae": validation_mae,
        }
    }


def _predict_non_negative(model, x_data):
    preds = np.asarray(model.predict(x_data), dtype=float)
    return np.maximum(preds, 0.0)


def _select_best_fitted_model(model_results, candidates):
    selected = {k: v for k, v in model_results.items() if k in candidates}
    if len(selected) == 0:
        raise ValueError("No model results available for best-model selection.")
    ranking = sorted(selected.items(), key=lambda item: (item[1]["metrics"]["mae"], item[1]["metrics"]["rmse"]))
    return ranking[0]


def _split_train_for_calibration(x_train, y_train):
    n_rows = len(x_train)
    split_idx = int(np.floor(n_rows * (1 - CALIBRATION_VALIDATION_SIZE)))
    split_idx = min(max(split_idx, 1), n_rows - 1)
    x_base = x_train.iloc[:split_idx].copy()
    y_base = y_train.iloc[:split_idx].copy()
    x_cal = x_train.iloc[split_idx:].copy()
    y_cal = y_train.iloc[split_idx:].copy()
    return x_base, y_base, x_cal, y_cal


def _fit_default_calibrator(y_true, y_pred):
    linear_model = fit_linear_calibrator(y_true=y_true, y_pred=y_pred)
    return {
        "calibration_type": "linear",
        "calibration_object": linear_model,
        "calibration_params": linear_calibrator_params(linear_model),
    }


def _apply_calibrator(y_pred, calibration_type, calibration_object):
    if calibration_type == "linear":
        return apply_linear_calibrator(y_pred=y_pred, model_or_params=calibration_object)
    if calibration_type == "multiplicative":
        return apply_multiplicative_calibrator(y_pred=y_pred, k=calibration_object)
    return np.asarray(y_pred, dtype=float)


def _build_raw_and_calibrated_results(model_name, train_fn, x_train, y_train, x_test, y_test, tune_rf, tune_xgb, use_calibration):
    final_model = train_fn(x_train, y_train, tune_rf, tune_xgb)
    raw_test_pred = _predict_non_negative(final_model, x_test)
    raw_metrics = compute_metrics(y_test, raw_test_pred)

    results = {
        f"{model_name}_raw": {
            "model": final_model,
            "model_base_name": model_name,
            "metrics": raw_metrics,
            "predictions": raw_test_pred,
            "calibration_enabled": False,
            "calibration_type": None,
            "calibration_params": None,
            "raw_metrics": raw_metrics,
            "calibrated_metrics": None,
        }
    }

    if not use_calibration or len(x_train) < 8:
        return results

    x_base, y_base, x_cal, y_cal = _split_train_for_calibration(x_train, y_train)
    base_model = train_fn(x_base, y_base, tune_rf, tune_xgb)
    cal_pred = _predict_non_negative(base_model, x_cal)

    linear_payload = _fit_default_calibrator(y_true=y_cal, y_pred=cal_pred)
    linear_cal_test_pred = _apply_calibrator(
        y_pred=raw_test_pred,
        calibration_type=linear_payload["calibration_type"],
        calibration_object=linear_payload["calibration_object"],
    )
    linear_metrics = compute_metrics(y_test, linear_cal_test_pred)

    mult_k = fit_multiplicative_calibrator(y_true=y_cal, y_pred=cal_pred)
    mult_cal_test_pred = apply_multiplicative_calibrator(y_pred=raw_test_pred, k=mult_k)
    mult_metrics = compute_metrics(y_test, mult_cal_test_pred)

    selected_type = "linear"
    selected_object = linear_payload["calibration_object"]
    selected_params = linear_payload["calibration_params"]
    selected_pred = linear_cal_test_pred
    selected_metrics = linear_metrics

    if mult_metrics["mae"] < linear_metrics["mae"]:
        selected_type = "multiplicative"
        selected_object = mult_k
        selected_params = {"k": float(mult_k)}
        selected_pred = mult_cal_test_pred
        selected_metrics = mult_metrics

    results[f"{model_name}_calibrated"] = {
        "model": final_model,
        "model_base_name": model_name,
        "metrics": selected_metrics,
        "predictions": selected_pred,
        "calibration_enabled": selected_metrics["mae"] < raw_metrics["mae"],
        "calibration_type": selected_type,
        "calibration_params": selected_params,
        "calibration_object": selected_object,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": selected_metrics,
    }

    if selected_metrics["mae"] >= raw_metrics["mae"]:
        results[f"{model_name}_calibrated"]["predictions"] = raw_test_pred
        results[f"{model_name}_calibrated"]["metrics"] = raw_metrics
        results[f"{model_name}_calibrated"]["calibration_enabled"] = False
        results[f"{model_name}_calibrated"]["calibration_type"] = None
        results[f"{model_name}_calibrated"]["calibration_params"] = None
        results[f"{model_name}_calibrated"]["calibration_object"] = None

    return results


def _build_log_results(model_name, x_train, y_train, x_test, y_test, tune_rf, tune_xgb):
    y_train_log = np.log1p(np.maximum(y_train.to_numpy(dtype=float), 0.0))
    log_model = _fit_model(model_name=model_name, x_train=x_train, y_train=y_train_log, tune_rf=tune_rf, tune_xgb=tune_xgb)
    wrapped_model = LogTargetModel(base_model=log_model)
    pred = _predict_non_negative(wrapped_model, x_test)
    metrics = compute_metrics(y_test, pred)
    return {
        f"{model_name}_log": {
            "model": wrapped_model,
            "model_base_name": model_name,
            "metrics": metrics,
            "predictions": pred,
            "calibration_enabled": False,
            "calibration_type": None,
            "calibration_params": None,
            "raw_metrics": metrics,
            "calibrated_metrics": None,
            "log_transform_used": True,
        }
    }


def _compute_sample_weights(y_train, alpha=WEIGHT_ALPHA_DEFAULT):
    values = y_train.to_numpy(dtype=float)
    mean_y = float(np.mean(values)) if len(values) > 0 else 0.0
    std_y = float(np.std(values)) if len(values) > 0 else 0.0
    if not np.isfinite(std_y) or std_y <= 1e-9:
        return np.ones(shape=len(values), dtype=float)
    return 1.0 + float(alpha) * np.abs(values - mean_y) / std_y


def _build_weighted_log_xgboost_results(x_train, y_train, x_test, y_test, tune_xgb, alpha=WEIGHT_ALPHA_DEFAULT):
    sample_weights = _compute_sample_weights(y_train=y_train, alpha=alpha)
    y_train_log = np.log1p(np.maximum(y_train.to_numpy(dtype=float), 0.0))
    model = train_xgboost(x_train=x_train, y_train=y_train_log, tune=tune_xgb, sample_weight=sample_weights)
    wrapped = LogTargetModel(base_model=model)
    pred = _predict_non_negative(wrapped, x_test)
    metrics = compute_metrics(y_test, pred)
    return {
        "xgboost_log_weighted": {
            "model": wrapped,
            "model_base_name": "xgboost",
            "metrics": metrics,
            "predictions": pred,
            "calibration_enabled": False,
            "calibration_type": None,
            "calibration_params": None,
            "raw_metrics": metrics,
            "calibrated_metrics": None,
            "log_transform_used": True,
            "weighted_training_used": True,
            "alpha_value": float(alpha),
        }
    }


def _plot_minimal_feature_results(results_df):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_df = results_df.copy()
    labels = plot_df["feature_set"].tolist()
    mae_values = plot_df["mae"].to_numpy(dtype=float)
    counts = plot_df["feature_count"].to_numpy(dtype=int)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, mae_values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE")
    plt.xlabel("Feature Set")
    plt.tight_layout()
    for idx, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), str(int(counts[idx])), ha="center", va="bottom")
    plt.savefig(MINIMAL_PLOT_PATH, dpi=150)
    plt.close()


def _build_detailed_test_export(match_level_df, predictions_df, best_model_name, best_model_payload):
    identifier_columns = [
        "match_id",
        "match_date",
        "season",
        "competition_name",
        "stage",
        "away_team",
        "matchday",
        "weekday_name",
        "kickoff_time_local",
        "kickoff_hour",
        "month",
    ]
    feature_columns = [
        "is_weekend",
        "is_midweek",
        "is_public_holiday",
        "is_school_holiday_flanders",
        "seasonpass_holders",
        "pct_free_tickets",
        "has_promotion",
        "promo_tickets_total",
        "weather_temp_mean_c",
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
        "ohl_interest",
        "num_articles",
        "avg_days_to_match",
        "attendance_last_match",
        "attendance_last_3_avg",
        "opponent_strength_score",
        "opponent_frequency_seen",
        "opponent_tier",
    ]
    prediction_columns_order = [
        "pred_mean_baseline",
        "pred_opponent_mean_baseline",
        "pred_linear_regression_raw",
        "pred_linear_regression_calibrated",
        "pred_random_forest_raw",
        "pred_random_forest_calibrated",
        "pred_xgboost_raw",
        "pred_xgboost_calibrated",
        "pred_weighted_blend_raw",
        "pred_ensemble_raw",
        "pred_ensemble_calibrated",
    ]
    error_columns_order = [
        "abs_error_mean_baseline",
        "abs_error_opponent_mean_baseline",
        "abs_error_linear_regression_raw",
        "abs_error_linear_regression_calibrated",
        "abs_error_random_forest_raw",
        "abs_error_random_forest_calibrated",
        "abs_error_xgboost_raw",
        "abs_error_xgboost_calibrated",
        "abs_error_weighted_blend_raw",
        "abs_error_ensemble_raw",
        "abs_error_ensemble_calibrated",
    ]

    source_columns = ["match_id"]
    source_columns.extend([c for c in identifier_columns if c != "match_id" and c in match_level_df.columns])
    source_columns.extend([c for c in feature_columns if c in match_level_df.columns])
    source_columns = list(dict.fromkeys(source_columns))

    source_df = match_level_df[source_columns].drop_duplicates(subset="match_id", keep="first").copy()
    detailed_df = predictions_df.merge(source_df, on="match_id", how="left", sort=False, suffixes=("", "_real"))

    for col in ["match_date", "away_team"]:
        real_col = f"{col}_real"
        if real_col in detailed_df.columns:
            detailed_df[col] = detailed_df[real_col]
            detailed_df = detailed_df.drop(columns=[real_col])

    best_prediction_column = f"pred_{best_model_name}"
    best_error_column = f"abs_error_{best_model_name}"
    detailed_df["best_model_name"] = best_model_name
    detailed_df["prediction_best_model"] = detailed_df[best_prediction_column] if best_prediction_column in detailed_df.columns else np.nan
    detailed_df["abs_error_best_model"] = detailed_df[best_error_column] if best_error_column in detailed_df.columns else np.nan

    calibration_enabled = bool(best_model_payload.get("calibration_enabled", False))
    if calibration_enabled:
        model_base_name = str(best_model_payload.get("model_base_name", ""))
        raw_column = f"pred_{model_base_name}_raw"
        calibrated_column = f"pred_{model_base_name}_calibrated"
        detailed_df["raw_prediction_best_model"] = detailed_df[raw_column] if raw_column in detailed_df.columns else np.nan
        if calibrated_column in detailed_df.columns:
            detailed_df["calibrated_prediction_best_model"] = detailed_df[calibrated_column]
        else:
            detailed_df["calibrated_prediction_best_model"] = detailed_df["prediction_best_model"]
        detailed_df["calibration_enabled"] = True
        detailed_df["calibration_type"] = best_model_payload.get("calibration_type")

    available_identifier_cols = [c for c in identifier_columns if c in detailed_df.columns]
    available_prediction_cols = [c for c in prediction_columns_order if c in detailed_df.columns]
    available_error_cols = [c for c in error_columns_order if c in detailed_df.columns]
    available_feature_cols = [c for c in feature_columns if c in detailed_df.columns]

    ordered_columns = []
    ordered_columns.extend(available_identifier_cols)
    if "actual" in detailed_df.columns:
        ordered_columns.append("actual")
    ordered_columns.extend(["best_model_name", "prediction_best_model", "abs_error_best_model"])
    ordered_columns.extend(available_prediction_cols)
    ordered_columns.extend(available_error_cols)

    if calibration_enabled:
        ordered_columns.extend(
            [
                "raw_prediction_best_model",
                "calibrated_prediction_best_model",
                "calibration_enabled",
                "calibration_type",
            ]
        )

    ordered_columns.extend(available_feature_cols)
    ordered_columns = list(dict.fromkeys([c for c in ordered_columns if c in detailed_df.columns]))
    return detailed_df[ordered_columns].copy()


def _build_error_analysis_summary(predictions_df, best_model_name):
    pred_col = f"pred_{best_model_name}"
    err_col = f"abs_error_{best_model_name}"
    if pred_col not in predictions_df.columns or err_col not in predictions_df.columns:
        return pd.DataFrame(columns=["section", "error_bucket", "count", "percentage", "opponent", "mae"])

    abs_error = predictions_df[err_col].to_numpy(dtype=float)
    buckets = np.where(abs_error < 500.0, "small", np.where(abs_error <= 1000.0, "medium", "large"))

    bucket_df = pd.DataFrame({"error_bucket": buckets})
    bucket_df = bucket_df.groupby("error_bucket", as_index=False).size().rename(columns={"size": "count"})
    total = int(bucket_df["count"].sum()) if len(bucket_df) > 0 else 0
    bucket_df["percentage"] = np.where(total > 0, bucket_df["count"] / total * 100.0, 0.0)
    bucket_df["section"] = "error_bucket"
    bucket_df["opponent"] = pd.NA
    bucket_df["mae"] = pd.NA

    opponent_df = pd.DataFrame(columns=["section", "error_bucket", "count", "percentage", "opponent", "mae"])
    if "away_team" in predictions_df.columns:
        grouped = predictions_df.groupby("away_team", as_index=False)[err_col].mean().rename(columns={err_col: "mae"})
        grouped["section"] = "mean_error_per_opponent"
        grouped["error_bucket"] = pd.NA
        grouped["count"] = pd.NA
        grouped["percentage"] = pd.NA
        grouped = grouped.rename(columns={"away_team": "opponent"})
        opponent_df = grouped[["section", "error_bucket", "count", "percentage", "opponent", "mae"]]

    worst_df = pd.DataFrame(columns=["section", "error_bucket", "count", "percentage", "opponent", "mae"])
    if len(opponent_df) > 0:
        worst_df = opponent_df.sort_values("mae", ascending=False).head(5).copy()
        worst_df["section"] = "worst_5_opponents_by_mae"

    records = bucket_df.to_dict(orient="records")
    if len(opponent_df) > 0:
        records.extend(opponent_df.to_dict(orient="records"))
    if len(worst_df) > 0:
        records.extend(worst_df.to_dict(orient="records"))
    return pd.DataFrame(records, columns=["section", "error_bucket", "count", "percentage", "opponent", "mae"])


def run_feature_minimization_experiment(dataset_df, tune_xgb=False):
    full_feature_columns = get_feature_columns(dataset_df)
    x_all, y_all, meta_all = split_features_target(dataset_df, full_feature_columns)
    x_train_all, x_test_all, y_train, y_test, _, _ = time_train_test_split(
        x=x_all,
        y=y_all,
        meta=meta_all,
        test_size=TEST_SIZE,
    )

    groups = get_feature_minimization_groups(dataset_df)
    rows = []

    for group_name, columns in groups.items():
        x_train_group = x_train_all[columns].copy()
        x_test_group = x_test_all[columns].copy()

        xgb_model = train_xgboost(x_train=x_train_group, y_train=y_train, tune=tune_xgb)
        xgb_pred = np.asarray(xgb_model.predict(x_test_group), dtype=float)
        xgb_pred = np.maximum(xgb_pred, 0.0)
        xgb_metrics = compute_metrics(y_test, xgb_pred)

        rf_model = train_random_forest(x_train=x_train_group, y_train=y_train, tune=False)
        rf_pred = np.asarray(rf_model.predict(x_test_group), dtype=float)
        rf_pred = np.maximum(rf_pred, 0.0)
        rf_metrics = compute_metrics(y_test, rf_pred)

        rows.append(
            {
                "feature_set": group_name,
                "feature_count": int(len(columns)),
                "features": "|".join(columns),
                "mae": float(xgb_metrics["mae"]),
                "rmse": float(xgb_metrics["rmse"]),
                "r2": float(xgb_metrics["r2"]),
                "mape": float(xgb_metrics["mape"]),
                "median_abs_error": float(xgb_metrics["median_abs_error"]),
                "rf_mae": float(rf_metrics["mae"]),
                "rf_rmse": float(rf_metrics["rmse"]),
                "rf_r2": float(rf_metrics["r2"]),
                "rf_mape": float(rf_metrics["mape"]),
                "rf_median_abs_error": float(rf_metrics["median_abs_error"]),
            }
        )

    import pandas as pd

    results_df = pd.DataFrame(rows).sort_values(["mae", "feature_count"], ascending=[True, True]).reset_index(drop=True)
    save_csv(results_df, MINIMAL_RESULTS_PATH)

    best_row = results_df.iloc[0]
    best_mae = float(best_row["mae"])
    acceptable_threshold = best_mae * 1.05
    acceptable_df = results_df[results_df["mae"] <= acceptable_threshold].copy()
    acceptable_df = acceptable_df.sort_values(["feature_count", "mae", "feature_set"], ascending=[True, True, True]).reset_index(drop=True)
    smallest_acceptable_row = acceptable_df.iloc[0]

    summary_lines = [
        "Feature Minimization Summary",
        f"Best feature set: {best_row['feature_set']}",
        f"Best feature count: {int(best_row['feature_count'])}",
        f"Best MAE: {best_mae:.4f}",
        f"Acceptable MAE threshold (5%): {acceptable_threshold:.4f}",
        f"Smallest acceptable feature set: {smallest_acceptable_row['feature_set']}",
        f"Smallest acceptable feature count: {int(smallest_acceptable_row['feature_count'])}",
        f"Smallest acceptable MAE: {float(smallest_acceptable_row['mae']):.4f}",
        f"Recommended deployment feature set: {smallest_acceptable_row['feature_set']}",
        "",
        "Feature sets ranked by MAE then feature count:",
    ]
    for _, row in results_df.iterrows():
        summary_lines.append(f"{row['feature_set']}: MAE={float(row['mae']):.4f}, features={int(row['feature_count'])}")

    MINIMAL_SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    _plot_minimal_feature_results(results_df)

    return {
        "results": results_df,
        "best_feature_set": str(best_row["feature_set"]),
        "smallest_acceptable_feature_set": str(smallest_acceptable_row["feature_set"]),
        "best_mae": best_mae,
        "acceptable_mae_threshold": acceptable_threshold,
    }


def run_pipeline(
    data_dir=DEFAULT_DATA_DIR,
    tune_rf=False,
    tune_xgb=False,
    tune_catboost=False,
    use_log_target=False,
    run_ablation=False,
    run_feature_minimization=False,
    use_calibration=False,
    use_weather_api=WEATHER_API_ENABLED_DEFAULT,
    use_external_transfermarkt=False,
):
    ensure_directories([OUTPUTS_DIR, PREDICTIONS_DIR, FEATURE_IMPORTANCE_DIR, MODELS_DIR, REPORTS_DIR])

    tables = load_raw_tables(data_dir)
    match_level_df, dataset_stats = build_match_level_dataset(tables, use_weather_api=use_weather_api, return_stats=True)

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

    internal_training_rows = int(len(x_train))
    internal_test_rows = int(len(x_test))
    external_training_rows = 0
    external_stats = {
        "enabled": bool(use_external_transfermarkt),
        "path": str(TRANSFERMARKT_EXTERNAL_PATH),
        "exists": Path(TRANSFERMARKT_EXTERNAL_PATH).exists(),
        "rows_loaded": 0,
        "rows_used": 0,
    }

    if use_external_transfermarkt:
        x_external, y_external, external_stats = load_external_transfermarkt_training_rows(
            external_csv_path=TRANSFERMARKT_EXTERNAL_PATH,
            feature_columns=feature_columns,
            fallback_target_mean=float(np.mean(y_train)) if len(y_train) > 0 else 0.0,
        )
        external_training_rows = int(len(x_external))
        if external_training_rows > 0:
            x_train = pd.concat([x_train.reset_index(drop=True), x_external.reset_index(drop=True)], ignore_index=True)
            y_train = pd.concat([y_train.reset_index(drop=True), y_external.reset_index(drop=True)], ignore_index=True)
        else:
            print(f"Warning: external Transfermarkt dataset is enabled but no usable rows were loaded from {TRANSFERMARKT_EXTERNAL_PATH}")

    feature_fill_values = build_feature_fill_values(x_train=x_train, feature_columns=feature_columns)

    global_mean = float(np.mean(y_train))
    pred_mean = mean_baseline(y_train=y_train, size=len(y_test))
    pred_opp_mean = opponent_mean_baseline(
        y_train=y_train,
        x_train=x_train,
        x_test=x_test,
        fallback_value=global_mean,
    )

    baseline_results = {
        "mean_baseline": {"metrics": compute_metrics(y_test, pred_mean), "predictions": pred_mean},
        "opponent_mean_baseline": {"metrics": compute_metrics(y_test, pred_opp_mean), "predictions": pred_opp_mean},
    }

    fitted_results = {}
    for base_name in MODEL_CANDIDATES:
        train_fn = lambda x_tr, y_tr, trf, txg, name=base_name: _fit_model(name, x_tr, y_tr, trf, txg)
        payloads = _build_raw_and_calibrated_results(
            model_name=base_name,
            train_fn=train_fn,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            tune_rf=tune_rf,
            tune_xgb=tune_xgb,
            use_calibration=use_calibration,
        )
        fitted_results.update(payloads)

        log_payload = _build_log_results(
            model_name=base_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            tune_rf=tune_rf,
            tune_xgb=tune_xgb,
        )
        fitted_results.update(log_payload)

    weighted_log_payload = _build_weighted_log_xgboost_results(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        tune_xgb=tune_xgb,
        alpha=WEIGHT_ALPHA_DEFAULT,
    )
    fitted_results.update(weighted_log_payload)

    weighted_blend_payloads = _build_weighted_blend_payload(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        tune_rf=tune_rf,
        tune_xgb=tune_xgb,
        use_calibration=use_calibration,
    )
    fitted_results.update(weighted_blend_payloads)

    all_results = {}
    all_results.update(baseline_results)
    for model_name, payload in fitted_results.items():
        all_results[model_name] = {"metrics": payload["metrics"], "predictions": payload["predictions"]}

    xgboost_candidates = {k: v for k, v in fitted_results.items() if k in ["xgboost_log", "xgboost_log_weighted"]}
    best_model_name, best_model_payload = _select_best_fitted_model(xgboost_candidates, list(xgboost_candidates.keys()))
    best_predictions = np.asarray(best_model_payload["predictions"], dtype=float)

    calibration_enabled_for_best = False
    calibration_type_for_best = None
    calibration_params_for_best = None
    raw_metrics_for_best = dict(best_model_payload["metrics"])
    calibrated_metrics_for_best = None

    if len(x_train) >= 8:
        x_base, y_base, x_cal, y_cal = _split_train_for_calibration(x_train, y_train)
        if best_model_name in ["xgboost_log", "xgboost_log_weighted"]:
            y_base_log = np.log1p(np.maximum(y_base.to_numpy(dtype=float), 0.0))
            cal_sample_weight = None
            if best_model_name == "xgboost_log_weighted":
                full_weights = _compute_sample_weights(y_train=y_train, alpha=WEIGHT_ALPHA_DEFAULT)
                cal_sample_weight = full_weights[: len(x_base)]
            cal_base_model = LogTargetModel(
                base_model=_fit_model(
                    "xgboost",
                    x_base,
                    y_base_log,
                    tune_rf=tune_rf,
                    tune_xgb=tune_xgb,
                    sample_weight=cal_sample_weight,
                )
            )
        else:
            cal_base_model = _fit_model("xgboost", x_base, y_base, tune_rf=tune_rf, tune_xgb=tune_xgb)

        cal_pred = _predict_non_negative(cal_base_model, x_cal)
        linear_model = fit_linear_calibrator(y_true=y_cal, y_pred=cal_pred)
        linear_params = linear_calibrator_params(linear_model)
        calibrated_test_pred = np.maximum(apply_linear_calibrator(y_pred=best_predictions, model_or_params=linear_params), 0.0)
        calibrated_test_metrics = compute_metrics(y_test, calibrated_test_pred)

        if calibrated_test_metrics["mae"] < raw_metrics_for_best["mae"]:
            calibration_enabled_for_best = True
            calibration_type_for_best = "linear"
            calibration_params_for_best = linear_params
            calibrated_metrics_for_best = calibrated_test_metrics
            best_predictions = calibrated_test_pred
            best_model_payload["predictions"] = calibrated_test_pred
            best_model_payload["metrics"] = calibrated_test_metrics
            calibrated_name = f"{best_model_name}_calibrated"
            fitted_results[calibrated_name] = {
                "model": best_model_payload["model"],
                "model_base_name": best_model_payload["model_base_name"],
                "metrics": calibrated_test_metrics,
                "predictions": calibrated_test_pred,
                "calibration_enabled": True,
                "calibration_type": "linear",
                "calibration_params": linear_params,
                "raw_metrics": raw_metrics_for_best,
                "calibrated_metrics": calibrated_test_metrics,
                "log_transform_used": bool(best_model_name == "xgboost_log"),
            }
            best_model_name = calibrated_name
            best_model_payload = fitted_results[calibrated_name]

    full_reference_columns = get_full_reference_feature_columns(match_level_df)
    mae_before_full = float("nan")
    mae_after_reduced = float(fitted_results["xgboost_raw"]["metrics"]["mae"])
    mae_delta = float("nan")
    mae_delta_pct = float("nan")
    if len(full_reference_columns) > len(feature_columns):
        x_full, y_full, meta_full = split_features_target(match_level_df, full_reference_columns)
        x_train_full, x_test_full, y_train_full, y_test_full, _, _ = time_train_test_split(
            x=x_full,
            y=y_full,
            meta=meta_full,
            test_size=TEST_SIZE,
        )
        xgb_full = train_xgboost(x_train=x_train_full, y_train=y_train_full, tune=tune_xgb)
        pred_full = np.asarray(xgb_full.predict(x_test_full), dtype=float)
        pred_full = np.maximum(pred_full, 0.0)
        mae_before_full = float(compute_metrics(y_test_full, pred_full)["mae"])
        mae_delta = mae_after_reduced - mae_before_full
        if np.isfinite(mae_before_full) and mae_before_full != 0:
            mae_delta_pct = (mae_delta / mae_before_full) * 100.0

    xgb_raw_mae = float(fitted_results.get("xgboost_raw", {}).get("metrics", {}).get("mae", np.nan))
    xgb_log_mae = float(fitted_results.get("xgboost_log", {}).get("metrics", {}).get("mae", np.nan))
    xgb_log_weighted_mae = float(fitted_results.get("xgboost_log_weighted", {}).get("metrics", {}).get("mae", np.nan))
    log_transform_improved_mae = bool(np.isfinite(xgb_raw_mae) and np.isfinite(xgb_log_mae) and xgb_log_mae < xgb_raw_mae)
    weighting_improved_mae = bool(np.isfinite(xgb_log_mae) and np.isfinite(xgb_log_weighted_mae) and xgb_log_weighted_mae < xgb_log_mae)

    all_results[best_model_name] = {
        "metrics": best_model_payload["metrics"],
        "predictions": best_model_payload["predictions"],
    }

    comparison_df = build_comparison_table(all_results)
    save_csv(comparison_df, OUTPUTS_DIR / "model_comparison.csv")

    predictions_dict = {
        "mean_baseline": pred_mean,
        "opponent_mean_baseline": pred_opp_mean,
    }
    for model_name, payload in fitted_results.items():
        predictions_dict[model_name] = payload["predictions"]
    predictions_dict[best_model_name] = best_predictions

    predictions_df = build_predictions_table(
        meta_test=meta_test,
        y_test=y_test,
        predictions_dict=predictions_dict,
    )
    save_csv(predictions_df, PREDICTIONS_DIR / "test_predictions.csv")

    detailed_test_predictions_df = _build_detailed_test_export(
        match_level_df=match_level_df,
        predictions_df=predictions_df,
        best_model_name=best_model_name,
        best_model_payload=best_model_payload,
    )
    save_csv(detailed_test_predictions_df, PREDICTIONS_DIR / "test_predictions_real_data_detailed.csv")

    error_column = f"abs_error_{best_model_name}"
    pred_column = f"pred_{best_model_name}"

    top_errors = predictions_df.sort_values(error_column, ascending=False).head(10).copy()
    top_errors["signed_error"] = top_errors["actual"] - top_errors[pred_column]
    top_errors["pct_error"] = np.where(
        top_errors["actual"] != 0,
        np.abs(top_errors["signed_error"]) / np.abs(top_errors["actual"]) * 100.0,
        np.nan,
    )
    save_csv(top_errors, PREDICTIONS_DIR / "top_error_cases.csv")

    error_analysis_df = _build_error_analysis_summary(predictions_df=predictions_df, best_model_name=best_model_name)
    save_csv(error_analysis_df, PREDICTIONS_DIR / "error_analysis_summary.csv")

    feature_model = best_model_payload["model"]
    feature_names, feature_values = get_model_feature_importance(feature_model)
    feature_importance_df = save_feature_importance(
        names=feature_names,
        values=feature_values,
        output_path=FEATURE_IMPORTANCE_DIR / "best_model_feature_importance.csv",
        top_n=30,
    )

    rf_for_inference = fitted_results["random_forest_raw"]["model"]
    xgb_for_inference = fitted_results["xgboost_raw"]["model"]
    joblib.dump(rf_for_inference, RANDOM_FOREST_ARTIFACT_PATH)
    joblib.dump(xgb_for_inference, XGBOOST_ARTIFACT_PATH)

    if best_model_payload["model_base_name"] == "ensemble":
        joblib.dump(
            {
                "type": "ensemble",
                "weights": ENSEMBLE_WEIGHTS,
                "component_models": {
                    "random_forest": str(RANDOM_FOREST_ARTIFACT_PATH),
                    "xgboost": str(XGBOOST_ARTIFACT_PATH),
                },
            },
            BEST_MODEL_ARTIFACT_PATH,
        )
    else:
        joblib.dump(best_model_payload["model"], BEST_MODEL_ARTIFACT_PATH)

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

    minimization_output = None
    if run_feature_minimization:
        minimization_output = run_feature_minimization_experiment(dataset_df=match_level_df, tune_xgb=tune_xgb)

    if not calibration_enabled_for_best:
        calibration_type_for_best = None
        calibration_params_for_best = None
        calibrated_metrics_for_best = None

    run_info = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "best_model_name": best_model_name,
        "best_model_base_name": best_model_payload["model_base_name"],
        "model_type": "log" if "_log" in best_model_name else "raw",
        "log_transform_used": bool("_log" in best_model_name),
        "weighted_training_used": bool("weighted" in best_model_name),
        "alpha_value": float(WEIGHT_ALPHA_DEFAULT),
        "primary_candidate": PRIMARY_MODEL_CANDIDATE,
        "tested_model_names": list(all_results.keys()),
        "rows_total": int(len(match_level_df)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "external_training_enabled": bool(use_external_transfermarkt),
        "external_training_rows": int(external_training_rows),
        "internal_training_rows": int(internal_training_rows),
        "internal_test_rows": int(internal_test_rows),
        "evaluation_dataset": "OH Leuven internal only",
        "external_dataset_stats": external_stats,
        "features_used": feature_columns,
        "user_input_features": ["match_date", "away_team", "stage", "kickoff_time"],
        "feature_fill_values": feature_fill_values,
        "use_weather_api": bool(use_weather_api),
        "weather_enrichment_stats": dataset_stats.get("weather", {}),
        "target_column": TARGET_COLUMN,
        "use_log_target": False,
        "metrics_by_model": {name: payload["metrics"] for name, payload in all_results.items()},
        "best_metrics": best_model_payload["metrics"],
        "data_dir": str(Path(data_dir)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "component_model_paths": {
            "random_forest": str(RANDOM_FOREST_ARTIFACT_PATH),
            "xgboost": str(XGBOOST_ARTIFACT_PATH),
        },
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "weighted_blend_weights": fitted_results.get("weighted_blend_raw", {}).get("component_weights", {}),
        "weighted_blend_validation_mae": fitted_results.get("weighted_blend_raw", {}).get("component_validation_mae", {}),
        "residual_model_path": None,
        "residual_correction_enabled": False,
        "calibration_enabled": calibration_enabled_for_best,
        "calibration_type": calibration_type_for_best,
        "raw_metrics_for_best_model": raw_metrics_for_best,
        "calibrated_metrics_for_best_model": calibrated_metrics_for_best,
        "calibration_training_method": "split train into base_train and calibration_validation, fit calibrator on validation predictions, retrain base model on full train",
        "calibration_params": calibration_params_for_best if calibration_enabled_for_best else None,
        "feature_reduction_comparison": {
            "xgboost_mae_before_full_reference": mae_before_full,
            "xgboost_mae_after_reduced": mae_after_reduced,
            "mae_delta_after_minus_before": mae_delta,
            "mae_delta_pct": mae_delta_pct,
            "full_reference_feature_count": len(full_reference_columns),
            "reduced_feature_count": len(feature_columns),
        },
        "xgboost_raw_mae": xgb_raw_mae,
        "xgboost_log_mae": xgb_log_mae,
        "weighted_model_mae": xgb_log_weighted_mae,
        "log_transform_improved_mae": log_transform_improved_mae,
        "weighting_improved_mae": weighting_improved_mae,
    }

    if minimization_output is not None:
        run_info["feature_minimization"] = {
            "best_feature_set": minimization_output["best_feature_set"],
            "smallest_acceptable_feature_set": minimization_output["smallest_acceptable_feature_set"],
            "best_mae": minimization_output["best_mae"],
            "acceptable_mae_threshold": minimization_output["acceptable_mae_threshold"],
            "results_path": str(MINIMAL_RESULTS_PATH),
            "summary_path": str(MINIMAL_SUMMARY_PATH),
        }

    with open(BEST_MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    external_summary = pd.DataFrame(
        [
            {
                "internal_training_rows": int(internal_training_rows),
                "external_training_rows": int(external_training_rows),
                "internal_test_rows": int(internal_test_rows),
                "best_model": best_model_name,
                "mae_internal_test": float(best_model_payload["metrics"]["mae"]),
                "rmse_internal_test": float(best_model_payload["metrics"]["rmse"]),
                "mape_internal_test": float(best_model_payload["metrics"]["mape"]),
                "used_external_transfermarkt": bool(use_external_transfermarkt and external_training_rows > 0),
            }
        ]
    )
    save_csv(external_summary, EXTERNAL_TRAINING_SUMMARY_PATH)

    summary_lines = [
        "Attendance Prediction Project Report",
        f"Best model: {best_model_name}",
        f"MAE: {best_model_payload['metrics']['mae']:.2f}",
        f"RMSE: {best_model_payload['metrics']['rmse']:.2f}",
        f"R2: {best_model_payload['metrics']['r2']:.4f}",
        f"MAPE: {best_model_payload['metrics']['mape']:.2f}",
        f"Median Abs Error: {best_model_payload['metrics']['median_abs_error']:.2f}",
        f"Predictions file: {PREDICTIONS_DIR / 'new_match_predictions.csv'}",
        f"Reduced features: {len(feature_columns)}",
        f"Weather API enabled: {bool(use_weather_api)}",
        "Residual correction enabled: False",
        f"External Transfermarkt enabled: {bool(use_external_transfermarkt)}",
        f"Internal train rows: {internal_training_rows}",
        f"External train rows: {external_training_rows}",
        f"Internal test rows: {internal_test_rows}",
        "Evaluation dataset: OH Leuven internal only",
    ]
    if isinstance(dataset_stats.get("weather"), dict):
        weather_stats = dataset_stats["weather"]
        summary_lines.append(f"Weather rows enriched: {int(weather_stats.get('rows_enriched', 0))}")
        summary_lines.append(f"Weather API failures: {int(weather_stats.get('api_failures', 0))}")
    if calibration_enabled_for_best:
        summary_lines.append(f"Calibration enabled: True")
        summary_lines.append(f"Calibration type: {calibration_type_for_best}")
        summary_lines.append(f"Raw MAE: {raw_metrics_for_best['mae']:.2f}")
        summary_lines.append(f"Calibrated MAE: {calibrated_metrics_for_best['mae']:.2f}")
    else:
        summary_lines.append("Calibration enabled: False")

    if minimization_output is not None:
        summary_lines.append(f"Feature minimization best set: {minimization_output['best_feature_set']}")
        summary_lines.append(f"Feature minimization smallest acceptable set: {minimization_output['smallest_acceptable_feature_set']}")

    if np.isfinite(mae_before_full):
        summary_lines.append(f"XGBoost MAE before reduction: {mae_before_full:.2f}")
        summary_lines.append(f"XGBoost MAE after reduction: {mae_after_reduced:.2f}")
        summary_lines.append(f"MAE delta: {mae_delta:.2f} ({mae_delta_pct:.2f}%)")

    if np.isfinite(xgb_log_mae):
        summary_lines.append(f"XGBoost log MAE: {xgb_log_mae:.2f}")
    if np.isfinite(xgb_log_weighted_mae):
        summary_lines.append(f"Weighted model MAE: {xgb_log_weighted_mae:.2f}")
        summary_lines.append(f"Weighting improved MAE: {weighting_improved_mae}")

    (REPORTS_DIR / "summary_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "dataset": match_level_df,
        "comparison": comparison_df,
        "predictions": predictions_df,
        "top_errors": top_errors,
        "error_analysis": error_analysis_df,
        "feature_importance": feature_importance_df,
        "ablation": None,
        "minimization": minimization_output,
        "run_info": run_info,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--tune-rf", action="store_true")
    parser.add_argument("--tune-xgb", action="store_true")
    parser.add_argument("--tune-catboost", action="store_true")
    parser.add_argument("--use-log-target", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-feature-minimization", action="store_true")
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--use-weather-api", action="store_true")
    parser.add_argument("--compare-weather-impact", action="store_true")
    parser.add_argument("--use-external-transfermarkt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.compare_weather_impact:
        comparison_result = run_weather_impact_comparison(
            data_dir=Path(args.data_dir),
            tune_rf=args.tune_rf,
            tune_xgb=args.tune_xgb,
            tune_catboost=args.tune_catboost,
            use_log_target=args.use_log_target,
            run_ablation=args.run_ablation,
            run_feature_minimization=args.run_feature_minimization,
            use_calibration=args.use_calibration,
        )
        print(comparison_result["comparison"].to_string(index=False))
        print(f"Saved weather impact comparison: {comparison_result['output_path']}")
        return

    result = run_pipeline(
        data_dir=Path(args.data_dir),
        tune_rf=args.tune_rf,
        tune_xgb=args.tune_xgb,
        tune_catboost=args.tune_catboost,
        use_log_target=args.use_log_target,
        run_ablation=args.run_ablation,
        run_feature_minimization=args.run_feature_minimization,
        use_calibration=args.use_calibration,
        use_weather_api=args.use_weather_api,
        use_external_transfermarkt=args.use_external_transfermarkt,
    )
    print(result["comparison"].to_string(index=False))
    print(f"Best model: {result['run_info']['best_model_name']}")
    reduction = result["run_info"].get("feature_reduction_comparison", {})
    before_mae = reduction.get("xgboost_mae_before_full_reference")
    after_mae = reduction.get("xgboost_mae_after_reduced")
    delta_pct = reduction.get("mae_delta_pct")
    if isinstance(before_mae, (int, float)) and np.isfinite(before_mae):
        print(f"XGBoost MAE before reduction: {before_mae:.2f}")
        print(f"XGBoost MAE after reduction: {after_mae:.2f}")
        print(f"MAE delta (%): {delta_pct:.2f}")
    if result["run_info"].get("calibration_enabled", False):
        print(f"Calibration type: {result['run_info'].get('calibration_type')}")
    print(f"Weather API enabled: {result['run_info'].get('use_weather_api', False)}")
    print(f"External Transfermarkt enabled: {result['run_info'].get('external_training_enabled', False)}")
    print(f"External training rows: {result['run_info'].get('external_training_rows', 0)}")
    print(f"Evaluation dataset: {result['run_info'].get('evaluation_dataset', 'OH Leuven internal only')}")
    if result["minimization"] is not None:
        print(f"Feature minimization best set: {result['minimization']['best_feature_set']}")
        print(f"Feature minimization smallest acceptable set: {result['minimization']['smallest_acceptable_feature_set']}")


if __name__ == "__main__":
    main()

