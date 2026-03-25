import numpy as np
import pandas as pd

from .config import BAD_WEATHER_RAIN_THRESHOLD, BAD_WEATHER_WIND_THRESHOLD, DATE_COLUMN, TARGET_COLUMN
from .external_data import build_transfermarkt_opponent_priors_from_df, merge_transfermarkt_opponent_priors
from .weather import WEATHER_COLUMNS, enrich_weather_for_matches


USER_INPUT_FEATURES = [DATE_COLUMN, "away_team", "stage", "kickoff_time"]

FEATURE_COLUMNS = [
    "stage",
    "competition_name",
    "away_team",
    "season",
    "matchday",
    "weekday_name",
    "is_weekend",
    "is_midweek",
    "is_public_holiday",
    "is_school_holiday_flanders",
    "kickoff_hour",
    "kickoff_minute",
    "kickoff_is_evening",
    "month",
    "quarter",
    "days_since_previous_home_match",
    "days_since_previous_match",
    "attendance_last_match",
    "attendance_last_3_avg",
    "attendance_last_5_avg",
    "attendance_rolling_std_3",
    "attendance_ewm",
    "attendance_last_same_weekday",
    "attendance_vs_same_opponent_last",
    "attendance_trend_3",
    "points_last_3_matches",
    "points_last_5_matches",
    "wins_last_5",
    "unbeaten_streak",
    "goals_scored_last_3_matches",
    "goals_conceded_last_3_matches",
    "goal_difference_last_5",
    "opponent_strength_score",
    "opponent_frequency_seen",
    "opponent_tier",
    "opponent_points_last_5",
    "opponent_goal_diff_last_5",
    "big_opponent_flag",
    "is_playoff_stage",
    "is_derby",
    "is_high_importance_match",
    "num_articles",
    "num_articles_3d",
    "num_articles_7d",
    "avg_days_to_match",
    "article_hype_score",
    "ohl_interest",
    "ohl_interest_3d_avg",
    "ohl_interest_7d_avg",
    "ohl_interest_trend",
    "weather_temp_mean_c",
    "weather_precipitation_mm",
    "weather_rain_mm",
    "weather_windspeed_max_kmh",
    "weather_bad_flag",
    "weather_stress_index",
    "tm_opponent_attendance_index",
    "tm_opponent_goal_diff_avg",
    "tm_opponent_points_avg",
    "tm_opponent_match_count",
    "tm_opponent_strength_tier",
]

FULL_REFERENCE_COLUMNS = FEATURE_COLUMNS.copy()

FEATURE_MINIMIZATION_GROUPS = {
    "minimal_input_derived": ["stage", "away_team", "weekday_name", "kickoff_hour", "month"],
    "minimal_plus_calendar": ["stage", "away_team", "weekday_name", "is_weekend", "is_midweek", "kickoff_hour", "month"],
    "minimal_plus_lag": [
        "stage",
        "away_team",
        "weekday_name",
        "is_weekend",
        "is_midweek",
        "kickoff_hour",
        "month",
        "attendance_last_match",
        "attendance_last_3_avg",
    ],
    "compact_reduced": [
        "stage",
        "away_team",
        "weekday_name",
        "kickoff_hour",
        "month",
        "attendance_last_match",
        "attendance_last_3_avg",
        "points_last_5_matches",
        "opponent_strength_score",
        "num_articles_3d",
        "ohl_interest_3d_avg",
        "weather_bad_flag",
        "tm_opponent_attendance_index",
    ],
    "full_current": FULL_REFERENCE_COLUMNS,
}


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _to_bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def _derive_season(date_series):
    year = date_series.dt.year
    month = date_series.dt.month
    start_year = np.where(month >= 7, year, year - 1)
    end_year = start_year + 1
    return pd.Series(start_year.astype(str) + "/" + end_year.astype(str), index=date_series.index)


def _parse_kickoff(series):
    values = series.fillna("").astype(str)
    parsed = pd.to_datetime(values, format="%H:%M:%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], format="%H:%M", errors="coerce")
    return parsed


def _prepare_match(df_match):
    df = df_match.copy()
    if "match_id" not in df.columns:
        df["match_id"] = [f"match_{i}" for i in range(len(df))]
    df["match_id"] = df["match_id"].astype(str)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN]).copy()
    df = df.drop_duplicates(subset="match_id")

    if "is_home_match" not in df.columns:
        df["is_home_match"] = True
    df["is_home_match"] = _to_bool_series(df["is_home_match"])

    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = np.nan
    df[TARGET_COLUMN] = _safe_numeric(df[TARGET_COLUMN])

    for col in ["result_home", "goals_home_ft", "goals_away_ft", "matchday"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = _safe_numeric(df[col]) if col != "result_home" else df[col].astype(str)

    kickoff_source = "kickoff_time_local" if "kickoff_time_local" in df.columns else "kickoff_time"
    if kickoff_source not in df.columns:
        df[kickoff_source] = "20:45:00"
    parsed = _parse_kickoff(df[kickoff_source])
    df["kickoff_hour"] = parsed.dt.hour.fillna(20).astype(int)
    df["kickoff_minute"] = parsed.dt.minute.fillna(0).astype(int)

    if "season" not in df.columns:
        df["season"] = _derive_season(df[DATE_COLUMN])
    else:
        df["season"] = df["season"].astype(str)

    if "stage" not in df.columns:
        df["stage"] = "unknown"
    if "away_team" not in df.columns:
        df["away_team"] = "unknown"
    if "competition_name" not in df.columns:
        df["competition_name"] = "unknown"

    df["month"] = df[DATE_COLUMN].dt.month.astype(int)
    df["quarter"] = df[DATE_COLUMN].dt.quarter.astype(int)
    return df.sort_values([DATE_COLUMN, "kickoff_hour", "kickoff_minute", "match_id"]).reset_index(drop=True)


def _prepare_context(df_context):
    if df_context is None or len(df_context) == 0:
        return pd.DataFrame(columns=["match_id"])
    df = df_context.copy()
    bool_cols = ["has_promotion", "is_weekend", "is_midweek", "is_public_holiday", "is_school_holiday_flanders"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = _to_bool_series(df[col]).astype(float)

    numeric_cols = [
        "weather_temp_mean_c",
        "weather_precipitation_mm",
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
        "weather_bad_flag",
        "promo_tickets_total",
        "pct_free_tickets",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _safe_numeric(df[col])

    keep_cols = [
        "match_id",
        "weekday_name",
        "is_weekend",
        "is_midweek",
        "is_public_holiday",
        "is_school_holiday_flanders",
        "weather_temp_mean_c",
        "weather_precipitation_mm",
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
        "weather_bad_flag",
        "has_promotion",
        "promo_tickets_total",
        "pct_free_tickets",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    if len(keep_cols) == 0:
        return pd.DataFrame(columns=["match_id"])
    return df[keep_cols].drop_duplicates(subset="match_id")


def _prepare_tickets(df_tickets):
    if df_tickets is None or len(df_tickets) == 0:
        return pd.DataFrame(columns=["match_id", "seasonpass_holders"])
    df = df_tickets.copy()
    if "seasonpass_holders" not in df.columns:
        df["seasonpass_holders"] = np.nan
    df["seasonpass_holders"] = _safe_numeric(df["seasonpass_holders"])
    return df[[c for c in ["match_id", "seasonpass_holders"] if c in df.columns]].drop_duplicates(subset="match_id")


def _prepare_trends(df_trends):
    if df_trends is None or len(df_trends) == 0 or "date" not in df_trends.columns or "ohl_interest" not in df_trends.columns:
        return pd.DataFrame(columns=["date", "ohl_interest", "ohl_interest_3d_avg", "ohl_interest_7d_avg", "ohl_interest_trend"])
    df = df_trends.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ohl_interest"] = _safe_numeric(df["ohl_interest"])
    df = df.dropna(subset=["date", "ohl_interest"]).copy()
    grouped = df.groupby("date", as_index=False)["ohl_interest"].mean().sort_values("date").reset_index(drop=True)
    grouped["ohl_interest_3d_avg"] = grouped["ohl_interest"].rolling(3, min_periods=1).mean()
    grouped["ohl_interest_7d_avg"] = grouped["ohl_interest"].rolling(7, min_periods=1).mean()
    grouped["ohl_interest_trend"] = grouped["ohl_interest_3d_avg"] - grouped["ohl_interest_7d_avg"]
    return grouped


def _prepare_articles(df_articles):
    if df_articles is None or len(df_articles) == 0 or "match_id" not in df_articles.columns:
        return pd.DataFrame(columns=["match_id", "num_articles", "avg_days_to_match", "num_articles_3d", "num_articles_7d", "article_hype_score"])

    df = df_articles.copy()
    df = df[df["match_id"].notna()].copy()
    if "article_id" not in df.columns:
        df["article_id"] = np.arange(len(df))
    if "days_to_match" not in df.columns:
        df["days_to_match"] = np.nan
    df["days_to_match"] = _safe_numeric(df["days_to_match"])
    df["days_to_match"] = df["days_to_match"].fillna(0.0)

    grouped = df.groupby("match_id", as_index=False).agg(
        num_articles=("article_id", "count"),
        avg_days_to_match=("days_to_match", "mean"),
        num_articles_3d=("days_to_match", lambda x: float((x <= 3).sum())),
        num_articles_7d=("days_to_match", lambda x: float((x <= 7).sum())),
    )
    grouped["article_hype_score"] = grouped["num_articles_3d"] / (1.0 + grouped["avg_days_to_match"].clip(lower=0.0))
    return grouped


def _compute_interest_feature(match_df, trends_df):
    out = match_df[["match_id", DATE_COLUMN]].copy()
    out["lookup_date"] = out[DATE_COLUMN].dt.normalize() - pd.Timedelta(days=1)
    if len(trends_df) == 0:
        out["ohl_interest"] = np.nan
        out["ohl_interest_3d_avg"] = np.nan
        out["ohl_interest_7d_avg"] = np.nan
        out["ohl_interest_trend"] = np.nan
        return out[["match_id", "ohl_interest", "ohl_interest_3d_avg", "ohl_interest_7d_avg", "ohl_interest_trend"]]

    merged = pd.merge_asof(
        out.sort_values("lookup_date"),
        trends_df.sort_values("date"),
        left_on="lookup_date",
        right_on="date",
        direction="backward",
    )
    return merged[["match_id", "ohl_interest", "ohl_interest_3d_avg", "ohl_interest_7d_avg", "ohl_interest_trend"]]


def _compute_match_state_features(df):
    rows = []
    home_history = []
    home_dates = []
    home_weekday_history = {}
    home_opponent_history = {}
    opponent_attendance_history = {}
    opponent_goal_diff_history = {}
    opponent_points_history = {}
    points_history = []
    wins_history = []
    goals_for_history = []
    goals_against_history = []
    all_match_dates = []

    last_home_date = None
    last_any_date = None

    for _, row in df.iterrows():
        current_date = pd.to_datetime(row[DATE_COLUMN], errors="coerce")
        away_team = str(row.get("away_team", "unknown"))
        weekday_name = str(row.get("weekday_name", "unknown"))

        attendance_last_match = home_history[-1] if len(home_history) >= 1 else np.nan
        attendance_last_3_avg = float(np.mean(home_history[-3:])) if len(home_history) >= 1 else np.nan
        attendance_last_5_avg = float(np.mean(home_history[-5:])) if len(home_history) >= 1 else np.nan
        attendance_rolling_std_3 = float(np.std(home_history[-3:])) if len(home_history) >= 2 else np.nan
        attendance_ewm = float(pd.Series(home_history).ewm(span=5, adjust=False).mean().iloc[-1]) if len(home_history) >= 1 else np.nan
        attendance_last_same_weekday = home_weekday_history.get(weekday_name, [np.nan])[-1]
        attendance_last_same_opponent = home_opponent_history.get(away_team, [np.nan])[-1]
        attendance_trend_3 = attendance_last_3_avg - attendance_last_5_avg if pd.notna(attendance_last_3_avg) and pd.notna(attendance_last_5_avg) else np.nan

        points_last_3 = float(np.mean(points_history[-3:])) if len(points_history) >= 1 else np.nan
        points_last_5 = float(np.mean(points_history[-5:])) if len(points_history) >= 1 else np.nan
        wins_last_5 = float(np.mean(wins_history[-5:])) if len(wins_history) >= 1 else np.nan

        unbeaten_streak = 0
        for val in reversed(points_history):
            if val > 0:
                unbeaten_streak += 1
            else:
                break

        goals_scored_last_3 = float(np.mean(goals_for_history[-3:])) if len(goals_for_history) >= 1 else np.nan
        goals_conceded_last_3 = float(np.mean(goals_against_history[-3:])) if len(goals_against_history) >= 1 else np.nan
        goal_diff_last_5 = float(np.mean((np.array(goals_for_history[-5:]) - np.array(goals_against_history[-5:])))) if len(goals_for_history) >= 1 else np.nan

        opp_att = opponent_attendance_history.get(away_team, [])
        opp_goal_diff = opponent_goal_diff_history.get(away_team, [])
        opp_points = opponent_points_history.get(away_team, [])
        opponent_strength_score = float(np.mean(opp_att)) if len(opp_att) > 0 else (float(np.mean(home_history)) if len(home_history) > 0 else np.nan)
        opponent_frequency_seen = int(len(opp_att))
        opponent_points_last_5 = float(np.mean(opp_points[-5:])) if len(opp_points) > 0 else np.nan
        opponent_goal_diff_last_5 = float(np.mean(opp_goal_diff[-5:])) if len(opp_goal_diff) > 0 else np.nan

        known_scores = [float(np.mean(v)) for v in opponent_attendance_history.values() if len(v) > 0]
        if len(known_scores) >= 4 and pd.notna(opponent_strength_score):
            q1 = float(np.quantile(known_scores, 0.25))
            q3 = float(np.quantile(known_scores, 0.75))
            if opponent_strength_score <= q1:
                opponent_tier = "low"
            elif opponent_strength_score >= q3:
                opponent_tier = "high"
            else:
                opponent_tier = "medium"
        else:
            opponent_tier = "medium"

        big_opponent_flag = float(opponent_tier == "high")

        days_since_previous_home_match = (current_date - last_home_date).days if last_home_date is not None else np.nan
        days_since_previous_match = (current_date - last_any_date).days if last_any_date is not None else np.nan

        stage = str(row.get("stage", "")).lower()
        is_playoff_stage = float(any(k in stage for k in ["play", "final", "knockout"]))
        is_derby = float(any(k in away_team.lower() for k in ["anderslecht", "brugge", "genk", "standard", "antwerp", "gent"]))
        is_high_importance_match = float((big_opponent_flag > 0) or (is_playoff_stage > 0) or (is_derby > 0))

        rows.append(
            {
                "match_id": row["match_id"],
                "attendance_last_match": attendance_last_match,
                "attendance_last_3_avg": attendance_last_3_avg,
                "attendance_last_5_avg": attendance_last_5_avg,
                "attendance_rolling_std_3": attendance_rolling_std_3,
                "attendance_ewm": attendance_ewm,
                "attendance_last_same_weekday": attendance_last_same_weekday,
                "attendance_vs_same_opponent_last": attendance_last_same_opponent,
                "attendance_trend_3": attendance_trend_3,
                "points_last_3_matches": points_last_3,
                "points_last_5_matches": points_last_5,
                "wins_last_5": wins_last_5,
                "unbeaten_streak": float(unbeaten_streak),
                "goals_scored_last_3_matches": goals_scored_last_3,
                "goals_conceded_last_3_matches": goals_conceded_last_3,
                "goal_difference_last_5": goal_diff_last_5,
                "opponent_strength_score": opponent_strength_score,
                "opponent_frequency_seen": float(opponent_frequency_seen),
                "opponent_tier": opponent_tier,
                "opponent_points_last_5": opponent_points_last_5,
                "opponent_goal_diff_last_5": opponent_goal_diff_last_5,
                "big_opponent_flag": big_opponent_flag,
                "days_since_previous_home_match": days_since_previous_home_match,
                "days_since_previous_match": days_since_previous_match,
                "is_playoff_stage": is_playoff_stage,
                "is_derby": is_derby,
                "is_high_importance_match": is_high_importance_match,
            }
        )

        observed = bool(row.get("is_observed", True))
        if observed and pd.notna(current_date):
            last_any_date = current_date
            all_match_dates.append(current_date)

        if observed and bool(row.get("is_home_match", False)) and pd.notna(row.get(TARGET_COLUMN)):
            attendance = float(row[TARGET_COLUMN])
            home_history.append(attendance)
            home_dates.append(current_date)
            home_weekday_history.setdefault(weekday_name, []).append(attendance)
            home_opponent_history.setdefault(away_team, []).append(attendance)
            opponent_attendance_history.setdefault(away_team, []).append(attendance)
            last_home_date = current_date

            gf = float(_safe_numeric(pd.Series([row.get("goals_home_ft")])).iloc[0]) if pd.notna(row.get("goals_home_ft")) else np.nan
            ga = float(_safe_numeric(pd.Series([row.get("goals_away_ft")])).iloc[0]) if pd.notna(row.get("goals_away_ft")) else np.nan
            if pd.notna(gf) and pd.notna(ga):
                goal_diff = gf - ga
                points = 3.0 if goal_diff > 0 else 1.0 if goal_diff == 0 else 0.0
                win = 1.0 if goal_diff > 0 else 0.0
            else:
                result_home = str(row.get("result_home", "")).upper()
                points = 3.0 if result_home == "W" else 1.0 if result_home == "D" else 0.0
                win = 1.0 if result_home == "W" else 0.0
                if pd.isna(gf):
                    gf = np.nan
                if pd.isna(ga):
                    ga = np.nan
                goal_diff = gf - ga if pd.notna(gf) and pd.notna(ga) else np.nan

            points_history.append(points)
            wins_history.append(win)
            goals_for_history.append(float(gf) if pd.notna(gf) else 0.0)
            goals_against_history.append(float(ga) if pd.notna(ga) else 0.0)
            opponent_points_history.setdefault(away_team, []).append(points)
            if pd.notna(goal_diff):
                opponent_goal_diff_history.setdefault(away_team, []).append(float(goal_diff))

    return pd.DataFrame(rows)


def _infer_matchday(df):
    out = df.copy()
    out["matchday"] = _safe_numeric(out.get("matchday", np.nan))
    missing = out["matchday"].isna()
    if missing.any():
        inferred = out.groupby("season").cumcount() + 1
        out.loc[missing, "matchday"] = inferred.loc[missing]
    return out


def _add_calendar_features(df):
    out = df.copy()
    out["weekday_name"] = out[DATE_COLUMN].dt.day_name().fillna("unknown")
    weekday = out[DATE_COLUMN].dt.weekday
    out["is_weekend"] = weekday.isin([5, 6]).astype(float)
    out["is_midweek"] = weekday.isin([1, 2, 3]).astype(float)
    out["kickoff_is_evening"] = (out["kickoff_hour"] >= 18).astype(float)
    out["month"] = out[DATE_COLUMN].dt.month.astype(int)
    out["quarter"] = out[DATE_COLUMN].dt.quarter.astype(int)
    return out


def _refresh_weather_bad_flag(df):
    out = df.copy()
    for col in ["weather_rain_mm", "weather_windspeed_max_kmh", "weather_temp_mean_c", "weather_precipitation_mm"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = _safe_numeric(out[col])
    out["weather_bad_flag"] = ((out["weather_rain_mm"].fillna(0.0) > BAD_WEATHER_RAIN_THRESHOLD) | (out["weather_windspeed_max_kmh"].fillna(0.0) > BAD_WEATHER_WIND_THRESHOLD)).astype(float)
    out["weather_stress_index"] = out["weather_rain_mm"].fillna(0.0) + 0.25 * out["weather_windspeed_max_kmh"].fillna(0.0)
    return out


def _normalize_types(df):
    out = df.copy()
    categorical_cols = ["season", "stage", "competition_name", "away_team", "weekday_name", "opponent_tier", "tm_opponent_strength_tier"]
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].astype(str).replace({"nan": "unknown", "None": "unknown"})

    numeric_cols = [
        c
        for c in FEATURE_COLUMNS
        if c in out.columns and c not in {"stage", "competition_name", "away_team", "season", "weekday_name", "opponent_tier", "tm_opponent_strength_tier"}
    ]
    for col in numeric_cols:
        out[col] = _safe_numeric(out[col])
    return out


def _apply_weather_enrichment(df, use_weather_api=False, enrich_only_unobserved=False):
    out = df.copy()
    for col in WEATHER_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    existing_weather_cols = ["weather_temp_mean_c", "weather_precipitation_mm", "weather_rain_mm", "weather_windspeed_max_kmh"]
    existing_weather_mask = out[existing_weather_cols].notna().any(axis=1)

    target_mask = pd.Series(True, index=out.index)
    if enrich_only_unobserved and "is_observed" in out.columns:
        target_mask = ~_to_bool_series(out["is_observed"])

    weather_stats = {
        "enabled": bool(use_weather_api),
        "rows_requested": int(target_mask.sum()),
        "rows_enriched": 0,
        "filled_cells": 0,
        "historical_calls": 0,
        "forecast_calls": 0,
        "cache_hits": 0,
        "api_failures": 0,
    }

    if use_weather_api and target_mask.any():
        weather_input = out.loc[target_mask, ["match_id", DATE_COLUMN]].copy()
        if "is_observed" in out.columns:
            weather_input["is_observed"] = out.loc[target_mask, "is_observed"].values
        weather_df, api_stats = enrich_weather_for_matches(match_df=weather_input, date_column=DATE_COLUMN, use_weather_api=True)
        weather_stats.update(api_stats)
        if len(weather_df) > 0:
            out = out.merge(weather_df, on="match_id", how="left", suffixes=("", "_api"))
            for col in WEATHER_COLUMNS:
                api_col = f"{col}_api"
                if api_col in out.columns:
                    fill_mask = out[col].isna() & target_mask
                    api_values = _safe_numeric(out[api_col])
                    weather_stats["filled_cells"] += int((fill_mask & api_values.notna()).sum())
                    out.loc[fill_mask, col] = api_values.loc[fill_mask]
                    out = out.drop(columns=[api_col])
            if "weather_source_api" in out.columns:
                out["weather_source"] = out.get("weather_source", "").astype(str)
                out["weather_source"] = out["weather_source"].replace({"nan": ""})
                out.loc[out["weather_source"].eq(""), "weather_source"] = out.loc[out["weather_source"].eq(""), "weather_source_api"].astype(str)
                out = out.drop(columns=["weather_source_api"])

    if "weather_source" not in out.columns:
        out["weather_source"] = ""
    out.loc[existing_weather_mask, "weather_source"] = "source_table"
    out = _refresh_weather_bad_flag(out)

    enriched_mask = out[existing_weather_cols].notna().any(axis=1)
    if target_mask.any():
        weather_stats["rows_enriched"] = int((enriched_mask & target_mask).sum())
    return out, weather_stats


def _build_feature_table(match_df, context_df, tickets_df, trends_df, articles_df, transfermarkt_df):
    merged = match_df.merge(context_df, on="match_id", how="left")
    merged = merged.merge(tickets_df, on="match_id", how="left")
    merged = merged.merge(articles_df, on="match_id", how="left")
    merged = merged.merge(_compute_interest_feature(match_df, trends_df), on="match_id", how="left")
    merged = _add_calendar_features(merged)
    merged = _infer_matchday(merged)

    state_features = _compute_match_state_features(merged.sort_values([DATE_COLUMN, "kickoff_hour", "kickoff_minute", "match_id"]))
    merged = merged.merge(state_features, on="match_id", how="left")

    tm_priors = build_transfermarkt_opponent_priors_from_df(transfermarkt_df, min_matches=2)
    merged = merge_transfermarkt_opponent_priors(merged, tm_priors)

    merged = _refresh_weather_bad_flag(merged)
    merged = _normalize_types(merged)
    merged = merged.sort_values([DATE_COLUMN, "kickoff_hour", "kickoff_minute", "match_id"]).reset_index(drop=True)
    return merged, {"opponents_with_priors": int(len(tm_priors))}


def _apply_inference_fallbacks(df):
    out = df.copy()
    observed_mask = _to_bool_series(out.get("is_observed", pd.Series(True, index=out.index))) & _to_bool_series(out.get("is_home_match", pd.Series(True, index=out.index))) & out[TARGET_COLUMN].notna()
    observed_vals = _safe_numeric(out.loc[observed_mask, TARGET_COLUMN])
    global_mean = float(observed_vals.mean()) if observed_vals.notna().any() else 0.0

    inference_mask = ~_to_bool_series(out.get("is_observed", pd.Series(True, index=out.index)))
    lag_like_cols = [
        "attendance_last_match",
        "attendance_last_3_avg",
        "attendance_last_5_avg",
        "attendance_rolling_std_3",
        "attendance_ewm",
        "attendance_last_same_weekday",
        "attendance_vs_same_opponent_last",
        "attendance_trend_3",
        "opponent_strength_score",
        "opponent_points_last_5",
        "opponent_goal_diff_last_5",
    ]
    fallback_counts = {}
    for col in lag_like_cols:
        if col in out.columns:
            miss = inference_mask & out[col].isna()
            fallback_counts[col] = int(miss.sum())
            out.loc[miss, col] = global_mean

    return out, {"global_mean": global_mean, "fallback_counts": fallback_counts}


def build_match_level_dataset(tables, use_weather_api=False, return_stats=False):
    match_df = _prepare_match(tables["match"])
    context_df = _prepare_context(tables.get("context", pd.DataFrame()))
    tickets_df = _prepare_tickets(tables.get("tickets", pd.DataFrame()))
    trends_df = _prepare_trends(tables.get("trends", pd.DataFrame()))
    articles_df = _prepare_articles(tables.get("articles", pd.DataFrame()))
    transfermarkt_df = tables.get("transfermarkt", pd.DataFrame())

    match_df["is_observed"] = True
    merged, tm_stats = _build_feature_table(match_df, context_df, tickets_df, trends_df, articles_df, transfermarkt_df)
    merged, weather_stats = _apply_weather_enrichment(merged, use_weather_api=use_weather_api, enrich_only_unobserved=False)
    merged = merged[merged["is_home_match"]].copy()
    merged = merged.dropna(subset=[TARGET_COLUMN]).copy()
    merged = merged.sort_values([DATE_COLUMN, "kickoff_hour", "kickoff_minute", "match_id"]).reset_index(drop=True)

    if return_stats:
        return merged, {"weather": weather_stats, "transfermarkt": tm_stats}
    return merged


def build_inference_dataset(tables, new_matches_df, return_stats=False, use_weather_api=False):
    historical = _prepare_match(tables["match"])
    context_df = _prepare_context(tables.get("context", pd.DataFrame()))
    tickets_df = _prepare_tickets(tables.get("tickets", pd.DataFrame()))
    trends_df = _prepare_trends(tables.get("trends", pd.DataFrame()))
    articles_df = _prepare_articles(tables.get("articles", pd.DataFrame()))
    transfermarkt_df = tables.get("transfermarkt", pd.DataFrame())

    historical["is_observed"] = True

    incoming = new_matches_df.copy()
    missing_required = [col for col in USER_INPUT_FEATURES if col not in incoming.columns]
    if missing_required:
        raise ValueError(f"Missing required input columns: {', '.join(missing_required)}")

    incoming[DATE_COLUMN] = pd.to_datetime(incoming[DATE_COLUMN], errors="coerce")
    if incoming[DATE_COLUMN].isna().any():
        raise ValueError("Some rows in input file have invalid match_date values")

    if "match_id" not in incoming.columns:
        incoming["match_id"] = [f"new_match_{i}" for i in range(len(incoming))]
    incoming["match_id"] = incoming["match_id"].astype(str)

    incoming["kickoff_time_local"] = incoming["kickoff_time"].astype(str)
    incoming["is_home_match"] = True
    incoming[TARGET_COLUMN] = np.nan
    incoming["season"] = _derive_season(incoming[DATE_COLUMN])
    incoming["is_observed"] = False

    union_cols = sorted(set(historical.columns).union(set(incoming.columns)))
    combined = pd.concat([historical.reindex(columns=union_cols), incoming.reindex(columns=union_cols)], ignore_index=True)

    merged, tm_stats = _build_feature_table(combined, context_df, tickets_df, trends_df, articles_df, transfermarkt_df)
    merged, weather_stats = _apply_weather_enrichment(merged, use_weather_api=use_weather_api, enrich_only_unobserved=True)
    merged, fallback_stats = _apply_inference_fallbacks(merged)

    inference_only = merged[~_to_bool_series(merged["is_observed"])].copy()
    inference_only = inference_only.sort_values([DATE_COLUMN, "kickoff_hour", "kickoff_minute", "match_id"]).reset_index(drop=True)

    if return_stats:
        return inference_only, {
            "user_input_features": USER_INPUT_FEATURES,
            "auto_generated_features": [c for c in FEATURE_COLUMNS if c not in USER_INPUT_FEATURES],
            "fallback": fallback_stats,
            "weather": weather_stats,
            "transfermarkt": tm_stats,
        }
    return inference_only


def get_feature_columns(df):
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def get_full_reference_feature_columns(df):
    return [col for col in FULL_REFERENCE_COLUMNS if col in df.columns]


def get_feature_minimization_groups(df):
    available = set(df.columns)
    groups = {}
    for name, cols in FEATURE_MINIMIZATION_GROUPS.items():
        present = [c for c in cols if c in available]
        if len(present) > 0:
            groups[name] = present
    return groups


def get_feature_group_columns(df, group_name):
    groups = get_feature_minimization_groups(df)
    if group_name not in groups:
        raise ValueError(f"Unknown or unavailable feature group: {group_name}")
    return groups[group_name]


def split_features_target(df, feature_columns):
    x = df[feature_columns].copy()
    y = _safe_numeric(df[TARGET_COLUMN])
    valid = y.notna()
    x = x.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    meta_cols = [c for c in ["match_id", DATE_COLUMN, "away_team"] if c in df.columns]
    meta = df.loc[valid, meta_cols].reset_index(drop=True)
    return x, y, meta


def time_train_test_split(x, y, meta, test_size):
    n_rows = len(x)
    split_idx = int(np.floor(n_rows * (1 - test_size)))
    split_idx = min(max(split_idx, 1), n_rows - 1)
    x_train = x.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    meta_train = meta.iloc[:split_idx].copy()
    x_test = x.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    meta_test = meta.iloc[split_idx:].copy()
    return x_train, x_test, y_train, y_test, meta_train, meta_test


def build_feature_fill_values(x_train, feature_columns):
    fill_values = {}
    for col in feature_columns:
        if col not in x_train.columns:
            continue
        series = x_train[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric = _safe_numeric(series)
            fill_values[col] = float(numeric.median()) if numeric.notna().any() else 0.0
        else:
            non_null = series.dropna().astype(str)
            fill_values[col] = str(non_null.mode().iloc[0]) if len(non_null) > 0 else "unknown"
    return fill_values


def prepare_inference_features(df, feature_columns, fill_values=None):
    out = _normalize_types(df.copy())
    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan

    for col in feature_columns:
        series = out[col]
        if pd.api.types.is_numeric_dtype(series):
            fallback = float(fill_values.get(col, 0.0)) if isinstance(fill_values, dict) else 0.0
            numeric = _safe_numeric(series)
            if not np.isfinite(fallback):
                fallback = float(numeric.median()) if numeric.notna().any() else 0.0
            out[col] = numeric.fillna(fallback)
        else:
            fallback = str(fill_values.get(col, "unknown")) if isinstance(fill_values, dict) else "unknown"
            out[col] = series.astype(str).replace({"nan": fallback, "None": fallback}).fillna(fallback)

    return out[feature_columns].copy()
