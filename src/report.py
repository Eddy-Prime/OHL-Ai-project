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
        f"Primary candidate: {metadata.get('primary_candidate', 'xgboost')}",
        f"Log target used: {bool(metadata.get('use_log_target', False))}",
        f"Validation folds: {metadata.get('validation_folds', 4)}",
        f"Best MAE: {best_model_row['mae']:.2f}",
        f"Best RMSE: {best_model_row['rmse']:.2f}",
        f"Best R2: {best_model_row['r2']:.4f}",
        f"Best MAPE: {best_model_row.get('mape', float('nan')):.2f}",
        f"Best Median Abs Error: {best_model_row.get('median_abs_error', float('nan')):.2f}",
        "Top features:",
    ]

    for _, row in top_features.iterrows():
        summary_lines.append(f"- {row['feature']}: {row['importance']:.4f}")

    ablation_path = OUTPUTS_DIR / "ablation_results.csv"
    if ablation_path.exists():
        ablation_df = pd.read_csv(ablation_path)
        summary_lines.append("\nFeature Group Ablation Analysis:")
        for _, row in ablation_df.iterrows():
            summary_lines.append(f"  {row['feature_group']}: MAE={row['mae']:.2f} ({int(row['num_features'])} features)")

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

