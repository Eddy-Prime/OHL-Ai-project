from pathlib import Path
import argparse
import json
import numpy as np
from datetime import datetime, timezone

import joblib

from .baselines import mean_baseline, opponent_mean_baseline
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
from .features import build_match_level_dataset, get_feature_columns, split_features_target, time_train_test_split
from .models import get_model_feature_importance, train_linear_regression, train_random_forest, train_xgboost, SimpleEnsemble
from .utils import ensure_directories

MODEL_CANDIDATES = ["linear_regression", "random_forest", "xgboost"]
BEST_SELECTION_CANDIDATES = ["random_forest", "xgboost", "ensemble"]
ENSEMBLE_WEIGHTS = {"xgboost": 0.7, "random_forest": 0.3}
RANDOM_FOREST_ARTIFACT_PATH = MODELS_DIR / "random_forest_model.joblib"
XGBOOST_ARTIFACT_PATH = MODELS_DIR / "xgboost_model.joblib"


def _train_model_candidate(model_name, x_train, y_train, x_test, tune_rf, tune_xgb):
    if model_name == "linear_regression":
        model = train_linear_regression(x_train=x_train, y_train=y_train)
    elif model_name == "random_forest":
        model = train_random_forest(x_train=x_train, y_train=y_train, tune=tune_rf)
    elif model_name == "xgboost":
        model = train_xgboost(x_train=x_train, y_train=y_train, tune=tune_xgb)
    else:
        raise ValueError(f"Unknown model candidate: {model_name}")
    predictions = np.asarray(model.predict(x_test), dtype=float)
    predictions = np.maximum(predictions, 0.0)
    return model, predictions


def _select_best_fitted_model(model_results, candidates):
    selected = {k: v for k, v in model_results.items() if k in candidates}
    if len(selected) == 0:
        raise ValueError("No model results available for best-model selection.")
    ranking = sorted(selected.items(), key=lambda item: (item[1]["metrics"]["mae"], item[1]["metrics"]["rmse"]))
    return ranking[0]


def run_pipeline(
    data_dir=DEFAULT_DATA_DIR,
    tune_rf=False,
    tune_xgb=False,
    tune_catboost=False,
    use_log_target=False,
    run_ablation=False,
):
    ensure_directories([OUTPUTS_DIR, PREDICTIONS_DIR, FEATURE_IMPORTANCE_DIR, MODELS_DIR, REPORTS_DIR])

    tables = load_raw_tables(data_dir)
    match_level_df = build_match_level_dataset(tables)

    feature_columns = get_feature_columns(match_level_df)
    x, y, meta = split_features_target(match_level_df, feature_columns)

    if len(x) < 2:
        raise ValueError("Not enough matches to create train and test sets. At least 2 rows are required.")

    x_train, x_test, y_train, y_test, meta_train, meta_test = time_train_test_split(
        x=x,
        y=y,
        meta=meta,
        test_size=TEST_SIZE,
    )

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
    for model_name in MODEL_CANDIDATES:
        model, predictions = _train_model_candidate(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            tune_rf=tune_rf,
            tune_xgb=tune_xgb,
        )
        fitted_results[model_name] = {
            "model": model,
            "use_log_target": False,
            "metrics": compute_metrics(y_test, predictions),
            "predictions": predictions,
        }

    ensemble_model = SimpleEnsemble(
        models_dict={
            "xgboost": fitted_results["xgboost"]["model"],
            "random_forest": fitted_results["random_forest"]["model"],
        },
        weights_dict=ENSEMBLE_WEIGHTS,
    )
    ensemble_predictions = np.asarray(ensemble_model.predict(x_test), dtype=float)
    ensemble_predictions = np.maximum(ensemble_predictions, 0.0)
    fitted_results["ensemble"] = {
        "model": ensemble_model,
        "use_log_target": False,
        "metrics": compute_metrics(y_test, ensemble_predictions),
        "predictions": ensemble_predictions,
    }

    all_results = {}
    all_results.update(baseline_results)
    for model_name, payload in fitted_results.items():
        all_results[model_name] = {"metrics": payload["metrics"], "predictions": payload["predictions"]}

    best_model_name, best_model_payload = _select_best_fitted_model(fitted_results, BEST_SELECTION_CANDIDATES)
    best_predictions = np.asarray(best_model_payload["predictions"], dtype=float)

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
    if best_model_name == "ensemble":
        feature_model = fitted_results["xgboost"]["model"]
    feature_names, feature_values = get_model_feature_importance(feature_model)
    feature_importance_df = save_feature_importance(
        names=feature_names,
        values=feature_values,
        output_path=FEATURE_IMPORTANCE_DIR / "best_model_feature_importance.csv",
        top_n=30,
    )

    joblib.dump(fitted_results["random_forest"]["model"], RANDOM_FOREST_ARTIFACT_PATH)
    joblib.dump(fitted_results["xgboost"]["model"], XGBOOST_ARTIFACT_PATH)

    if best_model_name == "ensemble":
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

    run_info = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "best_model_name": best_model_name,
        "primary_candidate": PRIMARY_MODEL_CANDIDATE,
        "tested_model_names": list(all_results.keys()),
        "rows_total": int(len(match_level_df)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "features_used": feature_columns,
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
    ]
    (REPORTS_DIR / "summary_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "dataset": match_level_df,
        "comparison": comparison_df,
        "predictions": predictions_df,
        "top_errors": top_errors,
        "feature_importance": feature_importance_df,
        "ablation": None,
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
    )
    print(result["comparison"].to_string(index=False))
    print(f"Best model: {result['run_info']['best_model_name']}")


if __name__ == "__main__":
    main()

