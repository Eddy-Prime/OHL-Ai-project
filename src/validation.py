from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from .config import DATE_COLUMN, TARGET_COLUMN
from .evaluate import compute_metrics


def time_series_rolling_split(x, y, meta, n_folds=5, min_train_size=None, test_size_ratio=0.15):
    n_rows = len(x)
    if min_train_size is None:
        min_train_size = max(int(0.5 * n_rows), 10)

    folds = []
    for fold_idx in range(n_folds):
        total_before_test = int(n_rows * (1.0 - test_size_ratio))
        if fold_idx < n_folds - 1:
            train_end = min_train_size + int((total_before_test - min_train_size) * (fold_idx + 1) / n_folds)
        else:
            train_end = total_before_test

        if train_end >= n_rows:
            break
        if fold_idx > 0:
            last_fold_test_end = folds[-1]["test_indices"][-1] if len(folds[-1]["test_indices"]) > 0 else train_end - 1
            if train_end <= last_fold_test_end + 1:
                break

        test_start = train_end
        test_end = min(train_end + max(int(n_rows * test_size_ratio), 2), n_rows)

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)

        if len(train_indices) < 5 or len(test_indices) < 2:
            continue

        train_dates = pd.to_datetime(meta.iloc[train_indices, meta.columns.tolist().index(DATE_COLUMN)], errors="coerce")
        test_dates = pd.to_datetime(meta.iloc[test_indices, meta.columns.tolist().index(DATE_COLUMN)], errors="coerce")

        if train_dates.isna().any() or test_dates.isna().any():
            continue
        if train_dates.max() > test_dates.min():
            continue

        folds.append({
            "fold_idx": fold_idx,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_date_range": (train_dates.min().date().isoformat(), train_dates.max().date().isoformat()),
            "test_date_range": (test_dates.min().date().isoformat(), test_dates.max().date().isoformat()),
        })

    return folds


def backtest_model(model_func, x, y, meta, folds, model_name="model"):
    results = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        train_idx = fold["train_indices"]
        test_idx = fold["test_indices"]

        x_train = x.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        x_test = x.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()
        meta_test = meta.iloc[test_idx].copy()

        try:
            model = model_func(x_train, y_train)
            preds = np.asarray(model.predict(x_test), dtype=float)
            preds = np.maximum(preds, 0.0)
        except Exception as e:
            continue

        metrics = compute_metrics(y_test, preds)
        metrics["fold_idx"] = fold_idx
        metrics["fold_train_count"] = len(train_idx)
        metrics["fold_test_count"] = len(test_idx)
        metrics["fold_train_dates"] = fold["train_date_range"]
        metrics["fold_test_dates"] = fold["test_date_range"]
        metrics["model_name"] = model_name

        results.append(metrics)

    if len(results) == 0:
        return pd.DataFrame(), {}

    results_df = pd.DataFrame(results)

    summary = {
        "model_name": model_name,
        "n_folds": len(results),
        "mae_mean": float(results_df["mae"].mean()),
        "mae_std": float(results_df["mae"].std()),
        "rmse_mean": float(results_df["rmse"].mean()),
        "rmse_std": float(results_df["rmse"].std()),
        "mape_mean": float(results_df["mape"].mean()),
        "mape_std": float(results_df["mape"].std()),
        "r2_mean": float(results_df["r2"].mean()),
        "r2_std": float(results_df["r2"].std()),
    }

    return results_df, summary


def summarize_backtest_results(all_backtest_results):
    rows = []
    for model_name, (results_df, summary) in all_backtest_results.items():
        if len(summary) > 0:
            rows.append(summary)

    if len(rows) == 0:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    return summary_df.sort_values(["mae_mean", "r2_mean"], ascending=[True, False]).reset_index(drop=True)


class RollingBacktestReport:
    def __init__(self):
        self.backtest_results = {}
        self.final_holdout_metrics = {}
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_backtest(self, model_name, results_df, summary_dict):
        self.backtest_results[model_name] = {"results_df": results_df, "summary": summary_dict}

    def add_final_holdout(self, model_name, metrics):
        self.final_holdout_metrics[model_name] = metrics

    def to_dict(self):
        return {
            "created_at": self.created_at,
            "backtest_results": {k: v["summary"] for k, v in self.backtest_results.items()},
            "final_holdout_metrics": self.final_holdout_metrics,
        }

    def save_to_json(self, path):
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_fold_details(self, path, model_name=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if model_name and model_name in self.backtest_results:
            results_df = self.backtest_results[model_name]["results_df"]
            results_df.to_csv(path, index=False)
        else:
            all_rows = []
            for model_name, data in self.backtest_results.items():
                df = data["results_df"].copy()
                all_rows.append(df)
            if len(all_rows) > 0:
                combined = pd.concat(all_rows, ignore_index=True)
                combined.to_csv(path, index=False)

