from pathlib import Path
import argparse
import json
import numpy as np
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
)
from .data_loader import load_raw_tables
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
from .models import get_model_feature_importance, train_linear_regression, train_random_forest, train_xgboost, SimpleEnsemble
from .utils import ensure_directories

MODEL_CANDIDATES = ["linear_regression", "random_forest", "xgboost"]
BEST_SELECTION_CANDIDATES = [
    "linear_regression_raw",
    "linear_regression_calibrated",
    "random_forest_raw",
    "random_forest_calibrated",
    "xgboost_raw",
    "xgboost_calibrated",
    "ensemble_raw",
    "ensemble_calibrated",
]
ENSEMBLE_WEIGHTS = {"xgboost": 0.7, "random_forest": 0.3}
RANDOM_FOREST_ARTIFACT_PATH = MODELS_DIR / "random_forest_model.joblib"
XGBOOST_ARTIFACT_PATH = MODELS_DIR / "xgboost_model.joblib"
MINIMAL_RESULTS_PATH = OUTPUTS_DIR / "minimal_feature_results.csv"
MINIMAL_SUMMARY_PATH = OUTPUTS_DIR / "minimal_feature_summary.txt"
MINIMAL_PLOT_PATH = OUTPUTS_DIR / "minimal_feature_plot.png"
CALIBRATION_VALIDATION_SIZE = 0.2


def _fit_model(model_name, x_train, y_train, tune_rf, tune_xgb):
    if model_name == "linear_regression":
        return train_linear_regression(x_train=x_train, y_train=y_train)
    if model_name == "random_forest":
        return train_random_forest(x_train=x_train, y_train=y_train, tune=tune_rf)
    if model_name == "xgboost":
        return train_xgboost(x_train=x_train, y_train=y_train, tune=tune_xgb)
    raise ValueError(f"Unknown model candidate: {model_name}")


def _fit_ensemble_model(x_train, y_train, tune_rf, tune_xgb):
    xgb_model = _fit_model("xgboost", x_train, y_train, tune_rf=tune_rf, tune_xgb=tune_xgb)
    rf_model = _fit_model("random_forest", x_train, y_train, tune_rf=tune_rf, tune_xgb=tune_xgb)
    return SimpleEnsemble(models_dict={"xgboost": xgb_model, "random_forest": rf_model}, weights_dict=ENSEMBLE_WEIGHTS)


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
):
    ensure_directories([OUTPUTS_DIR, PREDICTIONS_DIR, FEATURE_IMPORTANCE_DIR, MODELS_DIR, REPORTS_DIR])

    tables = load_raw_tables(data_dir)
    match_level_df = build_match_level_dataset(tables)

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

    ensemble_train_fn = lambda x_tr, y_tr, trf, txg: _fit_ensemble_model(x_tr, y_tr, trf, txg)
    ensemble_payloads = _build_raw_and_calibrated_results(
        model_name="ensemble",
        train_fn=ensemble_train_fn,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        tune_rf=tune_rf,
        tune_xgb=tune_xgb,
        use_calibration=use_calibration,
    )
    fitted_results.update(ensemble_payloads)

    all_results = {}
    all_results.update(baseline_results)
    for model_name, payload in fitted_results.items():
        all_results[model_name] = {"metrics": payload["metrics"], "predictions": payload["predictions"]}

    best_model_name, best_model_payload = _select_best_fitted_model(fitted_results, BEST_SELECTION_CANDIDATES)
    best_predictions = np.asarray(best_model_payload["predictions"], dtype=float)

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

    comparison_df = build_comparison_table(all_results)
    save_csv(comparison_df, OUTPUTS_DIR / "model_comparison.csv")

    predictions_dict = {
        "mean_baseline": pred_mean,
        "opponent_mean_baseline": pred_opp_mean,
    }
    for model_name, payload in fitted_results.items():
        predictions_dict[model_name] = payload["predictions"]

    predictions_df = build_predictions_table(
        meta_test=meta_test,
        y_test=y_test,
        predictions_dict=predictions_dict,
    )
    save_csv(predictions_df, PREDICTIONS_DIR / "test_predictions.csv")

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

    feature_model = best_model_payload["model"]
    if best_model_payload["model_base_name"] == "ensemble":
        if hasattr(feature_model, "models") and "xgboost" in feature_model.models:
            feature_model = feature_model.models["xgboost"]
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

    calibration_enabled_for_best = bool(best_model_payload.get("calibration_enabled", False))
    calibration_type_for_best = best_model_payload.get("calibration_type") if calibration_enabled_for_best else None
    raw_metrics_for_best = best_model_payload.get("raw_metrics")
    calibrated_metrics_for_best = best_model_payload.get("calibrated_metrics") if calibration_enabled_for_best else None

    run_info = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "best_model_name": best_model_name,
        "best_model_base_name": best_model_payload["model_base_name"],
        "primary_candidate": PRIMARY_MODEL_CANDIDATE,
        "tested_model_names": list(all_results.keys()),
        "rows_total": int(len(match_level_df)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "features_used": feature_columns,
        "user_input_features": ["match_date", "away_team", "stage", "kickoff_time"],
        "feature_fill_values": feature_fill_values,
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
        "calibration_enabled": calibration_enabled_for_best,
        "calibration_type": calibration_type_for_best,
        "raw_metrics_for_best_model": raw_metrics_for_best,
        "calibrated_metrics_for_best_model": calibrated_metrics_for_best,
        "calibration_training_method": "split train into base_train and calibration_validation, fit calibrator on validation predictions, retrain base model on full train",
        "calibration_params": best_model_payload.get("calibration_params") if calibration_enabled_for_best else None,
        "feature_reduction_comparison": {
            "xgboost_mae_before_full_reference": mae_before_full,
            "xgboost_mae_after_reduced": mae_after_reduced,
            "mae_delta_after_minus_before": mae_delta,
            "mae_delta_pct": mae_delta_pct,
            "full_reference_feature_count": len(full_reference_columns),
            "reduced_feature_count": len(feature_columns),
        },
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
    ]
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

    (REPORTS_DIR / "summary_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "dataset": match_level_df,
        "comparison": comparison_df,
        "predictions": predictions_df,
        "top_errors": top_errors,
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
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_pipeline(
        data_dir=Path(args.data_dir),
        tune_rf=args.tune_rf,
        tune_xgb=args.tune_xgb,
        tune_catboost=args.tune_catboost,
        use_log_target=args.use_log_target,
        run_ablation=args.run_ablation,
        run_feature_minimization=args.run_feature_minimization,
        use_calibration=args.use_calibration,
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
    if result["minimization"] is not None:
        print(f"Feature minimization best set: {result['minimization']['best_feature_set']}")
        print(f"Feature minimization smallest acceptable set: {result['minimization']['smallest_acceptable_feature_set']}")


if __name__ == "__main__":
    main()

