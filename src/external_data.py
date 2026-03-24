from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BAD_WEATHER_RAIN_THRESHOLD, BAD_WEATHER_WIND_THRESHOLD, DATE_COLUMN, TARGET_COLUMN


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _derive_season(date_series):
    year = date_series.dt.year
    month = date_series.dt.month
    start_year = np.where(month >= 7, year, year - 1)
    end_year = start_year + 1
    return pd.Series(start_year.astype(str) + "/" + end_year.astype(str), index=date_series.index)


def _parse_kickoff_hour(series):
    parsed = pd.to_datetime(series.astype(str), format="%H:%M:%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing].astype(str), format="%H:%M", errors="coerce")
    return parsed.dt.hour.fillna(18).astype(int)


def _compute_temporal_features(df, fallback_mean):
    history = []
    opponent_count = {}
    opponent_sum = {}
    rows = []

    for _, row in df.iterrows():
        attendance_last_match = history[-1] if len(history) > 0 else fallback_mean
        attendance_last_3_avg = float(np.mean(history[-3:])) if len(history) > 0 else fallback_mean

        opponent = str(row.get("away_team", "unknown"))
        seen = int(opponent_count.get(opponent, 0))
        if seen > 0:
            opp_strength = float(opponent_sum[opponent] / seen)
        else:
            opp_strength = float(np.mean(history)) if len(history) > 0 else fallback_mean

        known_scores = [opponent_sum[k] / opponent_count[k] for k in opponent_count if opponent_count[k] > 0]
        if pd.notna(opp_strength) and len(known_scores) >= 4:
            q1 = float(np.quantile(known_scores, 0.25))
            q3 = float(np.quantile(known_scores, 0.75))
            if opp_strength >= q3:
                opponent_tier = "high"
            elif opp_strength <= q1:
                opponent_tier = "low"
            else:
                opponent_tier = "medium"
        else:
            opponent_tier = "medium"

        rows.append(
            {
                "attendance_last_match": attendance_last_match,
                "attendance_last_3_avg": attendance_last_3_avg,
                "opponent_strength_score": opp_strength,
                "opponent_frequency_seen": seen,
                "opponent_tier": opponent_tier,
            }
        )

        target = row.get(TARGET_COLUMN)
        if pd.notna(target):
            val = float(target)
            history.append(val)
            opponent_count[opponent] = seen + 1
            opponent_sum[opponent] = float(opponent_sum.get(opponent, 0.0)) + val

    return pd.DataFrame(rows, index=df.index)


def load_external_transfermarkt_training_rows(external_csv_path, feature_columns, fallback_target_mean):
    path = Path(external_csv_path)
    if not path.exists():
        return pd.DataFrame(columns=feature_columns), pd.Series(dtype=float), {
            "enabled": False,
            "path": str(path),
            "exists": False,
            "rows_loaded": 0,
            "rows_used": 0,
        }

    try:
        raw = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame(columns=feature_columns), pd.Series(dtype=float), {
            "enabled": True,
            "path": str(path),
            "exists": True,
            "rows_loaded": 0,
            "rows_used": 0,
        }

    if len(raw) == 0:
        return pd.DataFrame(columns=feature_columns), pd.Series(dtype=float), {
            "enabled": True,
            "path": str(path),
            "exists": True,
            "rows_loaded": 0,
            "rows_used": 0,
        }

    df = raw.copy()

    if DATE_COLUMN not in df.columns:
        for candidate in ["date", "match_day", "match_datetime"]:
            if candidate in df.columns:
                df[DATE_COLUMN] = df[candidate]
                break
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if "away_team" not in df.columns:
        for candidate in ["opponent_name_normalized", "opponent_name_raw", "opponent"]:
            if candidate in df.columns:
                df["away_team"] = df[candidate]
                break
    if "away_team" not in df.columns:
        df["away_team"] = "unknown"

    if "stage" not in df.columns:
        df["stage"] = "unknown"
    if "competition_name" not in df.columns:
        df["competition_name"] = "unknown"

    if TARGET_COLUMN not in df.columns:
        if "attendance" in df.columns:
            df[TARGET_COLUMN] = df["attendance"]
        else:
            df[TARGET_COLUMN] = np.nan

    df[TARGET_COLUMN] = _safe_numeric(df[TARGET_COLUMN])
    df = df.dropna(subset=[DATE_COLUMN, TARGET_COLUMN]).copy()

    if "matchday" not in df.columns:
        df["matchday"] = np.nan
    df["matchday"] = _safe_numeric(df["matchday"])

    if "kickoff_time" not in df.columns:
        for candidate in ["kickoff_time_local", "time"]:
            if candidate in df.columns:
                df["kickoff_time"] = df[candidate]
                break
    if "kickoff_time" not in df.columns:
        df["kickoff_time"] = "18:00:00"

    df["kickoff_hour"] = _parse_kickoff_hour(df["kickoff_time"])
    df["month"] = df[DATE_COLUMN].dt.month.astype(int)
    df["weekday_name"] = df[DATE_COLUMN].dt.day_name().fillna("unknown")
    weekday = df[DATE_COLUMN].dt.weekday
    df["is_weekend"] = weekday.isin([5, 6]).astype(float)
    df["is_midweek"] = weekday.isin([1, 2, 3]).astype(float)

    if "season" not in df.columns:
        df["season"] = _derive_season(df[DATE_COLUMN])
    df["season"] = df["season"].astype(str)

    for col in [
        "weather_temp_mean_c",
        "weather_precipitation_mm",
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
        "seasonpass_holders",
        "promo_tickets_total",
        "pct_free_tickets",
        "num_articles",
        "avg_days_to_match",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = _safe_numeric(df[col])

    if "has_promotion" not in df.columns:
        df["has_promotion"] = 0.0
    df["has_promotion"] = _safe_numeric(df["has_promotion"]).fillna(0.0)

    if "is_public_holiday" not in df.columns:
        df["is_public_holiday"] = 0.0
    if "is_school_holiday_flanders" not in df.columns:
        df["is_school_holiday_flanders"] = 0.0

    rain = df["weather_rain_mm"].fillna(0.0)
    wind = df["weather_windspeed_max_kmh"].fillna(0.0)
    df["weather_bad_flag"] = ((rain > BAD_WEATHER_RAIN_THRESHOLD) | (wind > BAD_WEATHER_WIND_THRESHOLD)).astype(float)

    if "ohl_interest" not in df.columns:
        df["ohl_interest"] = np.nan

    df = df.sort_values([DATE_COLUMN, "kickoff_hour", "away_team"]).reset_index(drop=True)
    temporal = _compute_temporal_features(df, fallback_mean=float(fallback_target_mean))
    for col in temporal.columns:
        df[col] = temporal[col]

    if "opponent_tier" not in df.columns:
        df["opponent_tier"] = "medium"

    x_external = df.copy()
    for col in feature_columns:
        if col not in x_external.columns:
            x_external[col] = np.nan

    x_external = x_external[feature_columns].copy()
    y_external = _safe_numeric(df[TARGET_COLUMN]).reset_index(drop=True)
    valid = y_external.notna()
    x_external = x_external.loc[valid].reset_index(drop=True)
    y_external = y_external.loc[valid].reset_index(drop=True)

    stats = {
        "enabled": True,
        "path": str(path),
        "exists": True,
        "rows_loaded": int(len(raw)),
        "rows_used": int(len(x_external)),
    }
    return x_external, y_external, stats

