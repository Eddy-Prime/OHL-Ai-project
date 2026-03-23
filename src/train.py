from pathlib import Path
import argparse
import json
import numpy as np
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

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
    TRAINED_MODEL_CANDIDATES,
    N_SPLITS_VALIDATION,
    ENSEMBLE_WEIGHTS,
    SUPPORT_ENSEMBLE,
    FEATURE_GROUPS,
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
from .models import (
    get_model_feature_importance,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_catboost,
    SimpleEnsemble,
    is_catboost_available,
)
from .utils import ensure_directories

def _predict_with_optional_inverse(model, x_test, use_log_target):
    preds = np.asarray(model.predict(x_test), dtype=float)
    if use_log_target:
        preds = np.expm1(preds)
    return np.maximum(preds, 0.0)


def _train_model_candidate(model_name, x_train, y_train, x_test, tune_rf, tune_xgb, tune_catboost=False):
    if model_name == "linear_regression":
        model = train_linear_regression(x_train=x_train, y_train=y_train)
    elif model_name == "random_forest":
        model = train_random_forest(x_train=x_train, y_train=y_train, tune=tune_rf)
    elif model_name == "xgboost":
        model = train_xgboost(x_train=x_train, y_train=y_train, tune=tune_xgb)
    elif model_name == "catboost":
        model = train_catboost(x_train=x_train, y_train=y_train, tune=tune_catboost)
    else:
        raise ValueError(f"Unknown model candidate: {model_name}")
    predictions = np.asarray(model.predict(x_test), dtype=float)
    return model, predictions


def _create_walk_forward_splits(x, y, meta, n_splits=4):
    split_generator = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, test_idx in split_generator.split(x):
        x_train, x_val = x.iloc[train_idx].copy(), x.iloc[test_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        meta_train, meta_val = meta.iloc[train_idx].copy(), meta.iloc[test_idx].copy()
        folds.append({
            'x_train': x_train,
            'x_val': x_val,
            'y_train': y_train,
            'y_val': y_val,
            'meta_train': meta_train,
            'meta_val': meta_val,
        })
    return folds


def _run_ablation_analysis(x_train, y_train, x_test, y_test, feature_columns, tune_xgb):
    ablation_results = []
    
    cumulative_features = []
    for group_name in ['base', 'lag', 'form', 'opponent', 'schedule', 'media', 'trends', 'interactions']:
        group_features = FEATURE_GROUPS.get(group_name, [])
        if not group_features:
            continue
        
        cumulative_features.extend(group_features)
        valid_features = [f for f in cumulative_features if f in feature_columns]
        
        if len(valid_features) == 0:
            continue
        
        x_train_group = x_train[valid_features].copy()
        x_test_group = x_test[valid_features].copy()
        
        model, predictions = _train_model_candidate(
            model_name='xgboost',
            x_train=x_train_group,
            y_train=y_train,
            x_test=x_test_group,
            tune_rf=False,
            tune_xgb=tune_xgb,
            tune_catboost=False,
        )
        metrics = compute_metrics(y_test, predictions)
        
        ablation_results.append({
            'feature_group': group_name,
            'num_features': len(valid_features),
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
            'mape': metrics['mape'],
        })
    
    return pd.DataFrame(ablation_results)


def _select_best_fitted_model(model_results, mae_tie_threshold=5.0):
    ranking = sorted(model_results.items(), key=lambda item: (item[1]["metrics"]["mae"], item[1]["metrics"]["rmse"]))
    best_name, best_payload = ranking[0]
    if len(ranking) > 1:
        second_name, second_payload = ranking[1]
        mae_gap = abs(best_payload["metrics"]["mae"] - second_payload["metrics"]["mae"])
        if mae_gap <= mae_tie_threshold and second_payload["metrics"]["rmse"] < best_payload["metrics"]["rmse"]:
            best_name, best_payload = second_name, second_payload
    return best_name, best_payload


def _resolve_model_candidates(tune_catboost=False):
    candidates = []
    catboost_available = is_catboost_available()

    for model_name in TRAINED_MODEL_CANDIDATES:
        if model_name != "catboost":
            candidates.append(model_name)
            continue

        if catboost_available:
            candidates.append(model_name)
            continue

        if tune_catboost:
            raise ImportError("catboost is required when --tune-catboost is set. Install dependencies with: pip install -r requirements.txt")

        print("Skipping model candidate 'catboost' because the package is not installed.")

    return candidates


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

    active_model_candidates = _resolve_model_candidates(tune_catboost=tune_catboost)
    if len(active_model_candidates) == 0:
        raise ValueError("No trainable model candidates are available.")

    folds = _create_walk_forward_splits(x_train, y_train, meta_train, n_splits=N_SPLITS_VALIDATION)
    
    fitted_models_per_fold = {model_name: [] for model_name in active_model_candidates}
    fitted_results = {}
    
    for fold_idx, fold in enumerate(folds):
        x_fold_train = fold['x_train']
        y_fold_train = fold['y_train']
        
        for model_name in active_model_candidates:
            model, _ = _train_model_candidate(
                model_name=model_name,
                x_train=x_fold_train,
                y_train=y_fold_train,
                x_test=x_test,
                tune_rf=tune_rf,
                tune_xgb=tune_xgb,
                tune_catboost=tune_catboost,
            )
            fitted_models_per_fold[model_name].append(model)
    
    for model_name in active_model_candidates:
        if len(fitted_models_per_fold[model_name]) > 0:
            final_model = fitted_models_per_fold[model_name][-1]
            predictions = np.asarray(final_model.predict(x_test), dtype=float)
            fitted_results[model_name] = {
                "model": final_model,
                "use_log_target": False,
                "metrics": compute_metrics(y_test, predictions),
                "predictions": predictions,
            }
    
    if use_log_target:
        y_train_log = np.log1p(np.maximum(y_train.to_numpy(dtype=float), 0.0))
        fitted_models_log_per_fold = {model_name: [] for model_name in active_model_candidates}
        
        for fold_idx, fold in enumerate(folds):
            x_fold_train = fold['x_train']
            y_fold_train_log = np.log1p(np.maximum(fold['y_train'].to_numpy(dtype=float), 0.0))
            
            for model_name in active_model_candidates:
                model, _ = _train_model_candidate(
                    model_name=model_name,
                    x_train=x_fold_train,
                    y_train=y_fold_train_log,
                    x_test=x_test,
                    tune_rf=tune_rf,
                    tune_xgb=tune_xgb,
                    tune_catboost=tune_catboost,
                )
                fitted_models_log_per_fold[model_name].append(model)
        
        for model_name in active_model_candidates:
            if len(fitted_models_log_per_fold[model_name]) > 0:
                final_model = fitted_models_log_per_fold[model_name][-1]
                predictions = _predict_with_optional_inverse(final_model, x_test, use_log_target=True)
                fitted_results[f"{model_name}_log1p"] = {
                    "model": final_model,
                    "use_log_target": True,
                    "metrics": compute_metrics(y_test, predictions),
                    "predictions": predictions,
                }
    
    if SUPPORT_ENSEMBLE and "xgboost" in fitted_results and "catboost" in fitted_results:
        ensemble_models = {
            "xgboost": fitted_results["xgboost"]["model"],
            "catboost": fitted_results["catboost"]["model"],
        }
        ensemble = SimpleEnsemble(ensemble_models, ENSEMBLE_WEIGHTS)
        ensemble_predictions = np.asarray(ensemble.predict(x_test), dtype=float)
        fitted_results["ensemble"] = {
            "model": ensemble,
            "use_log_target": False,
            "metrics": compute_metrics(y_test, ensemble_predictions),
            "predictions": ensemble_predictions,
        }

    all_results = {}
    all_results.update(baseline_results)
    for model_name, payload in fitted_results.items():
        all_results[model_name] = {"metrics": payload["metrics"], "predictions": payload["predictions"]}

    best_model_name, best_model_payload = _select_best_fitted_model(fitted_results)
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
        np.nan
    )
    save_csv(top_errors, PREDICTIONS_DIR / "top_error_cases.csv")

    feature_names, feature_values = get_model_feature_importance(best_model_payload["model"])
    feature_importance_df = save_feature_importance(
        names=feature_names,
        values=feature_values,
        output_path=FEATURE_IMPORTANCE_DIR / "best_model_feature_importance.csv",
        top_n=30,
    )

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

    if run_ablation:
        ablation_df = _run_ablation_analysis(x_train, y_train, x_test, y_test, feature_columns, tune_xgb)
        save_csv(ablation_df, OUTPUTS_DIR / "ablation_results.csv")
    else:
        ablation_df = None

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
        "use_log_target": bool(best_model_payload["use_log_target"]),
        "validation_folds": N_SPLITS_VALIDATION,
        "metrics_by_model": {name: payload["metrics"] for name, payload in all_results.items()},
        "best_metrics": best_model_payload["metrics"],
        "data_dir": str(Path(data_dir)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
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
        f"Log target used: {bool(best_model_payload['use_log_target'])}",
        f"Walk-forward folds: {N_SPLITS_VALIDATION}",
        f"Predictions file: {PREDICTIONS_DIR / 'new_match_predictions.csv'}",
    ]
    (REPORTS_DIR / "summary_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "dataset": match_level_df,
        "comparison": comparison_df,
        "predictions": predictions_df,
        "top_errors": top_errors,
        "feature_importance": feature_importance_df,
        "ablation": ablation_df,
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
    if result["ablation"] is not None:
        print("\nAblation Results:")
        print(result["ablation"].to_string(index=False))


if __name__ == "__main__":
    main()



