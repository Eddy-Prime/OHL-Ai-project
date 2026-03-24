import argparse
import json

import pandas as pd

from .config import BEST_MODEL_METADATA_PATH, FEATURE_IMPORTANCE_DIR, NEW_MATCH_PREDICTIONS_PATH, OUTPUTS_DIR, PREDICTIONS_DIR, REPORTS_DIR
from .evaluate import plot_actual_vs_predicted, plot_attendance_over_time
from .utils import ensure_directories


def build_summary(regenerate_plots=False):
    ensure_directories([REPORTS_DIR])

    comparison_path = OUTPUTS_DIR / "model_comparison.csv"
    feature_path = FEATURE_IMPORTANCE_DIR / "best_model_feature_importance.csv"
    test_predictions_path = PREDICTIONS_DIR / "test_predictions.csv"
    new_predictions_path = NEW_MATCH_PREDICTIONS_PATH
    metadata_path = BEST_MODEL_METADATA_PATH

    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing file: {comparison_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing file: {feature_path}")
    if not test_predictions_path.exists():
        raise FileNotFoundError(f"Missing file: {test_predictions_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing file: {metadata_path}")

    comparison_df = pd.read_csv(comparison_path)
    feature_df = pd.read_csv(feature_path)
    test_predictions_df = pd.read_csv(test_predictions_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    best_model_name = metadata.get("best_model_name", comparison_df.sort_values("mae").iloc[0]["model"])
    best_model_row = comparison_df[comparison_df["model"] == best_model_name]
    if len(best_model_row) == 0:
        best_model_row = comparison_df.sort_values("mae").head(1)
    best_model_row = best_model_row.iloc[0]

    baseline_row = comparison_df[comparison_df["model"] == "mean_baseline"]
    baseline_mae = float(baseline_row.iloc[0]["mae"]) if len(baseline_row) > 0 else float("nan")
    mae_improvement = baseline_mae - float(best_model_row["mae"]) if pd.notna(baseline_mae) else float("nan")

    top_features = feature_df.head(5)
    prediction_column = f"pred_{best_model_name}"

    if regenerate_plots and prediction_column in test_predictions_df.columns and "actual" in test_predictions_df.columns:
        plot_actual_vs_predicted(
            y_true=test_predictions_df["actual"].to_numpy(),
            y_pred=test_predictions_df[prediction_column].to_numpy(),
            output_path=PREDICTIONS_DIR / "actual_vs_predicted_best_model.png",
        )
        if "match_date" in test_predictions_df.columns:
            plot_attendance_over_time(
                meta_test=test_predictions_df[["match_date"]].copy(),
                y_true=test_predictions_df["actual"].to_numpy(),
                y_pred=test_predictions_df[prediction_column].to_numpy(),
                output_path=PREDICTIONS_DIR / "attendance_over_time_test.png",
            )

    summary_lines = [
        "Attendance Prediction Project Report",
        f"Best model: {best_model_name}",
        f"Best MAE: {best_model_row['mae']:.2f}",
        f"Best RMSE: {best_model_row['rmse']:.2f}",
        f"Best R2: {best_model_row['r2']:.4f}",
        f"Best MAPE: {best_model_row.get('mape', float('nan')):.2f}",
        f"MAE improvement vs mean baseline: {mae_improvement:.2f}",
        f"Minimal required prediction input: {', '.join(metadata.get('user_input_features', ['match_date', 'away_team', 'stage', 'kickoff_time']))}",
        "Top features:",
    ]

    reduction = metadata.get("feature_reduction_comparison", {})
    before_mae = reduction.get("xgboost_mae_before_full_reference")
    after_mae = reduction.get("xgboost_mae_after_reduced")
    delta_pct = reduction.get("mae_delta_pct")
    if isinstance(before_mae, (int, float)) and pd.notna(before_mae):
        summary_lines.append(f"XGBoost MAE before reduction: {float(before_mae):.2f}")
        summary_lines.append(f"XGBoost MAE after reduction: {float(after_mae):.2f}")
        summary_lines.append(f"MAE delta after reduction (%): {float(delta_pct):.2f}")

    xgb_raw_mae = metadata.get("xgboost_raw_mae")
    xgb_log_mae = metadata.get("xgboost_log_mae")
    weighted_model_mae = metadata.get("weighted_model_mae")
    if isinstance(xgb_raw_mae, (int, float)) and isinstance(xgb_log_mae, (int, float)):
        summary_lines.append(f"XGBoost raw MAE: {float(xgb_raw_mae):.2f}")
        summary_lines.append(f"XGBoost log MAE: {float(xgb_log_mae):.2f}")
        summary_lines.append(f"Log-transform improved MAE: {bool(metadata.get('log_transform_improved_mae', False))}")
    if isinstance(weighted_model_mae, (int, float)):
        summary_lines.append(f"Weighted model MAE: {float(weighted_model_mae):.2f}")
        summary_lines.append(f"Weighting improved MAE: {bool(metadata.get('weighting_improved_mae', False))}")

    summary_lines.append(f"Model type: {metadata.get('model_type', 'raw')}")
    summary_lines.append(f"Log transform used: {bool(metadata.get('log_transform_used', False))}")

    weather_enabled = bool(metadata.get("use_weather_api", False))
    weather_stats = metadata.get("weather_enrichment_stats", {})
    summary_lines.append(f"Weather API enabled: {weather_enabled}")
    if isinstance(weather_stats, dict) and len(weather_stats) > 0:
        summary_lines.append(f"Weather rows enriched: {int(weather_stats.get('rows_enriched', 0))}")
        summary_lines.append(f"Weather API failures: {int(weather_stats.get('api_failures', 0))}")

    weather_impact_path = OUTPUTS_DIR / "weather_impact_comparison.csv"
    if weather_impact_path.exists():
        weather_impact_df = pd.read_csv(weather_impact_path)
        summary_lines.append(f"Weather impact comparison file: {weather_impact_path}")
        if len(weather_impact_df) > 0:
            best_model_weather_row = weather_impact_df[weather_impact_df["model"] == best_model_name]
            if len(best_model_weather_row) > 0:
                row = best_model_weather_row.iloc[0]
                summary_lines.append(
                    f"Weather impact for {best_model_name}: MAE without={float(row['mae_without_weather']):.2f}, with={float(row['mae_with_weather']):.2f}, delta%={float(row['delta_mae_pct']):.2f}"
                )

    summary_lines.append(f"External training enabled: {bool(metadata.get('external_training_enabled', False))}")
    summary_lines.append(f"Internal training rows: {int(metadata.get('internal_training_rows', metadata.get('rows_train', 0)))}")
    summary_lines.append(f"External training rows: {int(metadata.get('external_training_rows', 0))}")
    summary_lines.append(f"Internal test rows: {int(metadata.get('internal_test_rows', metadata.get('rows_test', 0)))}")
    summary_lines.append(f"Evaluation dataset: {metadata.get('evaluation_dataset', 'OH Leuven internal only')}")

    calibration_enabled = bool(metadata.get("calibration_enabled", False))
    calibration_type = metadata.get("calibration_type")
    raw_metrics = metadata.get("raw_metrics_for_best_model")
    calibrated_metrics = metadata.get("calibrated_metrics_for_best_model")
    summary_lines.append(f"Calibration enabled: {calibration_enabled}")
    if calibration_enabled:
        summary_lines.append(f"Calibration type: {calibration_type}")
        if isinstance(raw_metrics, dict) and isinstance(calibrated_metrics, dict):
            summary_lines.append(f"Raw MAE: {float(raw_metrics.get('mae', float('nan'))):.2f}")
            summary_lines.append(f"Calibrated MAE: {float(calibrated_metrics.get('mae', float('nan'))):.2f}")
        summary_lines.append("Calibration was applied to reduce systematic overprediction or underprediction bias")

    for _, row in top_features.iterrows():
        summary_lines.append(f"- {row['feature']}: {row['importance']:.4f}")

    if best_model_name == "ensemble":
        weights = metadata.get("ensemble_weights", {})
        summary_lines.append(f"Ensemble weights - xgboost: {float(weights.get('xgboost', 0.7)):.2f}, random_forest: {float(weights.get('random_forest', 0.3)):.2f}")

    minimal_results_path = OUTPUTS_DIR / "minimal_feature_results.csv"
    minimal_summary_path = OUTPUTS_DIR / "minimal_feature_summary.txt"
    if minimal_results_path.exists():
        minimal_df = pd.read_csv(minimal_results_path)
        if len(minimal_df) > 0:
            best_min_row = minimal_df.sort_values(["mae", "feature_count"], ascending=[True, True]).iloc[0]
            threshold = float(best_min_row["mae"]) * 1.05
            acceptable_df = minimal_df[minimal_df["mae"] <= threshold].copy()
            acceptable_df = acceptable_df.sort_values(["feature_count", "mae", "feature_set"], ascending=[True, True, True])
            smallest_acceptable = acceptable_df.iloc[0]

            summary_lines.append("Feature minimization:")
            summary_lines.append(f"- Best feature set: {best_min_row['feature_set']} (MAE={float(best_min_row['mae']):.4f}, features={int(best_min_row['feature_count'])})")
            summary_lines.append(f"- Smallest acceptable set: {smallest_acceptable['feature_set']} (MAE={float(smallest_acceptable['mae']):.4f}, features={int(smallest_acceptable['feature_count'])})")
            summary_lines.append("Feature minimization comparison:")
            for _, row in minimal_df.sort_values(["mae", "feature_count"]).iterrows():
                summary_lines.append(f"  {row['feature_set']}: MAE={float(row['mae']):.4f}, features={int(row['feature_count'])}")

    if minimal_summary_path.exists():
        summary_lines.append(f"Feature minimization summary file: {minimal_summary_path}")

    if new_predictions_path.exists():
        new_predictions_df = pd.read_csv(new_predictions_path)
        if "predicted_attendance" in new_predictions_df.columns and len(new_predictions_df) > 0:
            summary_lines.append(f"New match predictions rows: {len(new_predictions_df)}")
            summary_lines.append(f"New predictions mean attendance: {new_predictions_df['predicted_attendance'].mean():.2f}")
            summary_lines.append(f"Prediction file: {new_predictions_path}")
    else:
        summary_lines.append(f"Prediction file not found yet: {new_predictions_path}")

    report_path = REPORTS_DIR / "summary_report.txt"
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "summary_lines": summary_lines,
        "report_path": str(report_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_summary(regenerate_plots=args.regenerate_plots)
    print("\n".join(result["summary_lines"]))
    print(f"Saved: {result['report_path']}")


if __name__ == "__main__":
    main()

