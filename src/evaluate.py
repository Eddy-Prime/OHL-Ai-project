from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from .utils import safe_ratio


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_errors = np.abs(y_true - y_pred)
    non_zero_mask = np.abs(y_true) > 1e-9
    if non_zero_mask.any():
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
    else:
        mape = float("nan")
    r2_value = float("nan")
    if len(y_true) >= 2:
        r2_value = float(r2_score(y_true, y_pred))
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_value,
        "mape": mape,
        "median_abs_error": float(median_absolute_error(y_true, y_pred)),
        "max_abs_error": float(abs_errors.max()),
        "share_abs_error_le_500": safe_ratio((abs_errors <= 500).sum(), len(abs_errors)),
        "share_abs_error_le_800": safe_ratio((abs_errors <= 800).sum(), len(abs_errors)),
        "share_abs_error_le_1000": safe_ratio((abs_errors <= 1000).sum(), len(abs_errors)),
    }
    return metrics


def build_comparison_table(results):
    rows = []
    for model_name, payload in results.items():
        row = {"model": model_name}
        row.update(payload["metrics"])
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    return table


def build_predictions_table(meta_test, y_test, predictions_dict):
    out = meta_test.copy()
    out["actual"] = y_test.to_numpy(dtype=float)
    for model_name, preds in predictions_dict.items():
        out[f"pred_{model_name}"] = np.asarray(preds, dtype=float)
        out[f"abs_error_{model_name}"] = np.abs(out["actual"] - out[f"pred_{model_name}"])
    return out


def save_feature_importance(names, values, output_path, top_n=20):
    frame = pd.DataFrame({"feature": names, "importance": values})
    frame = frame.head(top_n).copy()
    frame.to_csv(output_path, index=False)
    return frame


def plot_actual_vs_predicted(y_true, y_pred, output_path):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    min_value = min(np.min(y_true), np.min(y_pred))
    max_value = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], color="red")
    plt.xlabel("Actual Attendance")
    plt.ylabel("Predicted Attendance")
    plt.title("Actual vs Predicted Attendance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(feature_importance_df, output_path):
    plot_df = feature_importance_df.sort_values("importance").tail(15)
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Importance")
    plt.title("Top Model Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_attendance_over_time(meta_test, y_true, y_pred, output_path):
    chart = meta_test.copy()
    chart["actual"] = np.asarray(y_true, dtype=float)
    chart["predicted"] = np.asarray(y_pred, dtype=float)
    chart["match_date"] = pd.to_datetime(chart["match_date"], errors="coerce")
    chart = chart.sort_values("match_date")
    plt.figure(figsize=(10, 5))
    plt.plot(chart["match_date"], chart["actual"], marker="o", label="Actual")
    plt.plot(chart["match_date"], chart["predicted"], marker="o", label="Predicted")
    plt.xlabel("Match Date")
    plt.ylabel("Attendance")
    plt.title("Test Set Attendance Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_attendance_distribution(y_values, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(np.asarray(y_values, dtype=float), bins=12)
    plt.xlabel("Attendance")
    plt.ylabel("Count")
    plt.title("Attendance Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_distribution(y_true, y_pred, output_path):
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=14)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_csv(df, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

