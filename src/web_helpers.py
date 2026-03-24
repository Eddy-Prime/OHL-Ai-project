from __future__ import annotations

from datetime import date, datetime, time, timezone
from pathlib import Path
import tempfile

import pandas as pd

from .config import BEST_MODEL_ARTIFACT_PATH, BEST_MODEL_METADATA_PATH, DEFAULT_DATA_DIR
from .data_loader import load_raw_tables
from .features import build_match_level_dataset
from .predict import load_metadata, predict_from_file


def derive_season_from_date(match_date: date) -> str:
    year = int(match_date.year)
    if int(match_date.month) >= 7:
        return f"{year}/{year + 1}"
    return f"{year - 1}/{year}"


def build_single_input_frame(match_date: date, away_team: str, stage: str, kickoff_time: time) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_date": match_date.isoformat(),
                "away_team": str(away_team),
                "stage": str(stage),
                "kickoff_time": kickoff_time.strftime("%H:%M:%S"),
            }
        ]
    )


def run_prediction_from_frame(
    input_df: pd.DataFrame,
    data_dir: Path = DEFAULT_DATA_DIR,
    model_path: Path = BEST_MODEL_ARTIFACT_PATH,
    metadata_path: Path = BEST_MODEL_METADATA_PATH,
    use_weather_api: bool = False,
):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as input_tmp:
        input_path = Path(input_tmp.name)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as output_tmp:
        output_path = Path(output_tmp.name)

    try:
        input_df.to_csv(input_path, index=False)
        result_df, summary = predict_from_file(
            input_file=input_path,
            data_dir=data_dir,
            model_path=model_path,
            metadata_path=metadata_path,
            output_file=output_path,
            use_weather_api=use_weather_api,
        )
        return result_df, summary
    finally:
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()


def load_historical_match_level(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    tables = load_raw_tables(data_dir)
    return build_match_level_dataset(tables)


def load_selector_options(data_dir: Path = DEFAULT_DATA_DIR):
    tables = load_raw_tables(data_dir)
    match_df = tables["match"].copy()

    if "is_home_match" in match_df.columns:
        mask = match_df["is_home_match"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])
        match_df = match_df[mask].copy()

    away_options = sorted([str(v) for v in match_df.get("away_team", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()])
    stage_options = sorted([str(v) for v in match_df.get("stage", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()])

    if len(away_options) == 0:
        away_options = ["unknown"]
    if len(stage_options) == 0:
        stage_options = ["unknown"]

    return away_options, stage_options


def compute_context_statistics(history_df: pd.DataFrame, match_date: date, away_team: str):
    df = history_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date", "tickets_scanned"]).sort_values("match_date")

    target_date = pd.to_datetime(match_date)
    prior = df[df["match_date"] < target_date].copy()

    season = derive_season_from_date(match_date)
    season_prior = prior[prior["season"].astype(str) == season].copy() if "season" in prior.columns else prior.copy()

    season_avg = float(season_prior["tickets_scanned"].mean()) if len(season_prior) > 0 else float(prior["tickets_scanned"].mean()) if len(prior) > 0 else float("nan")
    last_home = float(prior.iloc[-1]["tickets_scanned"]) if len(prior) > 0 else float("nan")
    last3_avg = float(prior.tail(3)["tickets_scanned"].mean()) if len(prior) > 0 else float("nan")

    same_opp = prior[prior["away_team"].astype(str) == str(away_team)].copy() if "away_team" in prior.columns else pd.DataFrame()
    same_opp_avg = float(same_opp["tickets_scanned"].mean()) if len(same_opp) > 0 else float("nan")

    recent_trend = prior.tail(10).copy()

    return {
        "season": season,
        "season_avg": season_avg,
        "last_home": last_home,
        "last3_avg": last3_avg,
        "same_opponent_avg": same_opp_avg,
        "recent_trend": recent_trend,
    }


def get_improvement_vs_baseline(metadata: dict) -> float:
    metrics = metadata.get("metrics_by_model", {})
    best = metadata.get("best_metrics", {})
    baseline = metrics.get("mean_baseline", {})
    baseline_mae = baseline.get("mae")
    best_mae = best.get("mae")
    if baseline_mae is None or best_mae is None:
        return float("nan")
    return float(baseline_mae) - float(best_mae)


def read_optional_csv(path: Path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


def format_int_like(value):
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.0f}"
    except Exception:
        return "N/A"


def load_project_metadata(metadata_path: Path = BEST_MODEL_METADATA_PATH):
    return load_metadata(metadata_path=metadata_path)


def build_interpretation(predicted: float, season_avg: float, last3_avg: float) -> str:
    if pd.isna(predicted):
        return "Prediction is not available."

    season_text = "close to"
    trend_text = "close to"

    if not pd.isna(season_avg):
        if predicted > season_avg * 1.05:
            season_text = "above"
        elif predicted < season_avg * 0.95:
            season_text = "below"

    if not pd.isna(last3_avg):
        if predicted > last3_avg * 1.05:
            trend_text = "above"
        elif predicted < last3_avg * 0.95:
            trend_text = "below"

    consistency = "consistent"
    if not pd.isna(season_avg) and not pd.isna(last3_avg):
        direction_match = (predicted >= season_avg and predicted >= last3_avg) or (predicted <= season_avg and predicted <= last3_avg)
        consistency = "consistent" if direction_match else "partly inconsistent"

    return f"This match is predicted {season_text} the season average, {trend_text} the recent home trend, and is {consistency} with recent patterns."


def build_prediction_export_frame(
    match_date: date,
    away_team: str,
    stage: str,
    kickoff_time: time,
    predicted: float,
    lower_mae: float,
    upper_mae: float,
    lower_med: float,
    upper_med: float,
    best_model: str,
):
    return pd.DataFrame(
        [
            {
                "match_date": match_date.isoformat(),
                "away_team": away_team,
                "stage": stage,
                "kickoff_time": kickoff_time.strftime("%H:%M:%S"),
                "best_model": best_model,
                "predicted_attendance": float(predicted),
                "lower_estimate_mae": float(lower_mae),
                "upper_estimate_mae": float(upper_mae),
                "lower_estimate_median_abs_error": float(lower_med),
                "upper_estimate_median_abs_error": float(upper_med),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )

