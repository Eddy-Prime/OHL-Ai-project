import numpy as np
import pandas as pd

from .config import BAD_WEATHER_RAIN_THRESHOLD, BAD_WEATHER_WIND_THRESHOLD, DATE_COLUMN, TARGET_COLUMN
from .weather import WEATHER_COLUMNS, enrich_weather_for_matches


USER_INPUT_FEATURES = [DATE_COLUMN, "away_team", "stage", "kickoff_time"]
AUTO_GENERATED_FEATURES = [
    "season",
    "matchday",
    "weekday_name",
    "is_weekend",
    "is_midweek",
    "kickoff_hour",
    "month",
    "attendance_last_match",
    "attendance_last_3_avg",
    "opponent_strength_score",
    "opponent_frequency_seen",
    "opponent_tier",
    "weather_temp_mean_c",
    "weather_precipitation_mm",
    "weather_rain_mm",
    "weather_windspeed_max_kmh",
    "weather_bad_flag",
]
REMOVABLE_FEATURES = [
    "weather_temp_mean_c",
    "weather_rain_mm",
    "weather_windspeed_max_kmh",
    "num_articles",
    "avg_days_to_match",
    "ohl_interest",
    "seasonpass_holders",
    "promo_tickets_total",
    "pct_free_tickets",
    "has_promotion",
]

FEATURE_COLUMNS = [
    "stage",
    "away_team",
    "season",
    "matchday",
    "weekday_name",
    "is_weekend",
    "is_midweek",
    "kickoff_hour",
    "month",
    "opponent_strength_score",
    "opponent_frequency_seen",
    "opponent_tier",
    "weather_temp_mean_c",
    "weather_precipitation_mm",
    "weather_rain_mm",
    "weather_windspeed_max_kmh",
    "weather_bad_flag",
    "attendance_last_match",
    "attendance_last_3_avg",
]

FULL_REFERENCE_COLUMNS = [
    "season",
    "stage",
    "competition_name",
    "away_team",
    "matchday",
    "weekday_name",
    "is_weekend",
    "is_midweek",
    "is_public_holiday",
    "is_school_holiday_flanders",
    "kickoff_hour",
    "month",
    "opponent_strength_score",
    "opponent_frequency_seen",
    "opponent_tier",
    "weather_temp_mean_c",
    "weather_precipitation_mm",
    "weather_rain_mm",
    "weather_windspeed_max_kmh",
    "weather_bad_flag",
    "seasonpass_holders",
    "promo_tickets_total",
    "pct_free_tickets",
    "has_promotion",
    "ohl_interest",
    "num_articles",
    "attendance_last_match",
    "attendance_last_3_avg",
    "avg_days_to_match",
]

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
    "compact_reduced": FEATURE_COLUMNS,
    "full_current": FULL_REFERENCE_COLUMNS,
}


def _to_bool_series(series):
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y", "t"])


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _derive_season(date_series):
    year = date_series.dt.year
    month = date_series.dt.month
    start_year = np.where(month >= 7, year, year - 1)
    end_year = start_year + 1
    return pd.Series(start_year.astype(str) + "-" + end_year.astype(str), index=date_series.index)


def _parse_kickoff_hour(series):
    values = series.astype(str)
    parsed = pd.to_datetime(values, format="%H:%M:%S", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(values.loc[remaining], format="%H:%M", errors="coerce")
    return parsed.dt.hour.fillna(0).astype(int)


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

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = _safe_numeric(df[TARGET_COLUMN])
    else:
        df[TARGET_COLUMN] = np.nan

    if "matchday" in df.columns:
        df["matchday"] = _safe_numeric(df["matchday"])
    else:
        df["matchday"] = np.nan

    kickoff_source = "kickoff_time_local" if "kickoff_time_local" in df.columns else "kickoff_time"
    if kickoff_source in df.columns:
        df["kickoff_hour"] = _parse_kickoff_hour(df[kickoff_source])
    else:
        df["kickoff_hour"] = 0

    df["month"] = df[DATE_COLUMN].dt.month.astype(int)

    if "season" not in df.columns:
        df["season"] = _derive_season(df[DATE_COLUMN])
    else:
        df["season"] = df["season"].astype(str)

    if "stage" not in df.columns:
        df["stage"] = "unknown"
    if "away_team" not in df.columns:
        df["away_team"] = "unknown"

    df = df.sort_values([DATE_COLUMN, "kickoff_hour", "match_id"]).reset_index(drop=True)
    return df


def _prepare_context(df_context):
    df = df_context.copy()
    bool_cols = ["has_promotion", "is_weekend", "is_midweek", "is_public_holiday", "is_school_holiday_flanders"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = _to_bool_series(df[col]).fillna(False)

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
    return df[keep_cols].drop_duplicates(subset="match_id") if keep_cols else pd.DataFrame({"match_id": []})


def _prepare_tickets(df_tickets):
    df = df_tickets.copy()
    if "seasonpass_holders" in df.columns:
        df["seasonpass_holders"] = _safe_numeric(df["seasonpass_holders"])
    keep_cols = [c for c in ["match_id", "seasonpass_holders"] if c in df.columns]
    return df[keep_cols].drop_duplicates(subset="match_id") if keep_cols else pd.DataFrame({"match_id": []})


def _prepare_trends(df_trends):
    df = df_trends.copy()
    if "date" not in df.columns or "ohl_interest" not in df.columns:
        return pd.DataFrame(columns=["date", "ohl_interest", "ohl_interest_3d_avg"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ohl_interest"] = _safe_numeric(df["ohl_interest"])
    df = df.dropna(subset=["date", "ohl_interest"]).copy()
    trend = df.groupby("date", as_index=False)["ohl_interest"].mean()
    trend = trend.sort_values("date").reset_index(drop=True)
    trend["ohl_interest_3d_avg"] = trend["ohl_interest"].rolling(window=3, min_periods=1).mean()
    return trend


def _prepare_articles(df_articles):
    df = df_articles.copy()
    if "match_id" not in df.columns:
        return pd.DataFrame(columns=["match_id", "num_articles", "avg_days_to_match"])
    df = df[df["match_id"].notna()].copy()
    if "article_id" not in df.columns:
        df["article_id"] = np.arange(len(df))
    df["days_to_match"] = _safe_numeric(df.get("days_to_match"))
    grouped = df.groupby("match_id", as_index=False).agg(
        num_articles=("article_id", "count"),
        avg_days_to_match=("days_to_match", "mean"),
    )
    return grouped


def _compute_interest_feature(match_df, trends_df):
    if len(trends_df) == 0:
        return pd.DataFrame({"match_id": match_df["match_id"], "ohl_interest": np.nan})
    out = match_df[["match_id", DATE_COLUMN]].copy()
    out["lookup_date"] = out[DATE_COLUMN].dt.normalize() - pd.Timedelta(days=1)
    merged = pd.merge_asof(
        out.sort_values("lookup_date"),
        trends_df.sort_values("date"),
        left_on="lookup_date",
        right_on="date",
        direction="backward",
    )
    merged["ohl_interest"] = merged["ohl_interest_3d_avg"]
    return merged[["match_id", "ohl_interest"]]


def _compute_lag_features(df):
    history = []
    opponent_count = {}
    opponent_sum = {}
    rows = []
    for _, row in df.iterrows():
        last_match = history[-1] if len(history) >= 1 else np.nan
        last_3_avg = float(np.mean(history[-3:])) if len(history) >= 1 else np.nan
        global_mean = float(np.mean(history)) if len(history) >= 1 else np.nan

        opponent = str(row.get("away_team", "unknown"))
        opp_seen = int(opponent_count.get(opponent, 0))
        if opp_seen > 0:
            opp_score = float(opponent_sum[opponent] / opp_seen)
        else:
            opp_score = global_mean

        known_scores = [opponent_sum[k] / opponent_count[k] for k in opponent_count if opponent_count[k] > 0]
        if pd.notna(opp_score) and len(known_scores) >= 4:
            q1 = float(np.quantile(known_scores, 0.25))
            q3 = float(np.quantile(known_scores, 0.75))
            if opp_score >= q3:
                opp_tier = "high"
            elif opp_score <= q1:
                opp_tier = "low"
            else:
                opp_tier = "medium"
        else:
            opp_tier = "medium"

        rows.append(
            {
                "match_id": row["match_id"],
                "attendance_last_match": last_match,
                "attendance_last_3_avg": last_3_avg,
                "opponent_strength_score": opp_score,
                "opponent_frequency_seen": opp_seen,
                "opponent_tier": opp_tier,
            }
        )
        if bool(row.get("is_observed", True)) and bool(row.get("is_home_match", False)) and pd.notna(row.get(TARGET_COLUMN)):
            attendance = float(row[TARGET_COLUMN])
            history.append(attendance)
            opponent_count[opponent] = opp_seen + 1
            opponent_sum[opponent] = float(opponent_sum.get(opponent, 0.0)) + attendance
    return pd.DataFrame(rows)


def _infer_matchday(df):
    out = df.copy()
    if "matchday" not in out.columns:
        out["matchday"] = np.nan
    out["matchday"] = _safe_numeric(out["matchday"])
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
    out["month"] = out[DATE_COLUMN].dt.month.astype(int)
    return out


def _normalize_types(df):
    out = df.copy()

    for col in ["season", "stage", "competition_name", "away_team", "weekday_name"]:
        if col in out.columns:
            out[col] = out[col].astype(str).replace({"nan": "unknown", "None": "unknown"})

    for col in ["is_weekend", "is_midweek", "is_public_holiday", "is_school_holiday_flanders", "has_promotion"]:
        if col in out.columns:
            out[col] = _to_bool_series(out[col]).astype(float)

    for col in [
        "matchday",
        "kickoff_hour",
        "month",
        "opponent_strength_score",
        "opponent_frequency_seen",
        "weather_temp_mean_c",
        "weather_precipitation_mm",
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
        "weather_bad_flag",
        "seasonpass_holders",
        "promo_tickets_total",
        "pct_free_tickets",
        "ohl_interest",
        "num_articles",
        "attendance_last_match",
        "attendance_last_3_avg",
        "avg_days_to_match",
    ]:
        if col in out.columns:
            out[col] = _safe_numeric(out[col])

    if "opponent_tier" in out.columns:
        out["opponent_tier"] = out["opponent_tier"].astype(str).replace({"nan": "medium", "None": "medium"})

    return out


def _refresh_weather_bad_flag(df):
    out = df.copy()
    for col in ["weather_rain_mm", "weather_windspeed_max_kmh"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = _safe_numeric(out[col])

    rain = out["weather_rain_mm"].fillna(0.0)
    wind = out["weather_windspeed_max_kmh"].fillna(0.0)
    out["weather_bad_flag"] = ((rain > BAD_WEATHER_RAIN_THRESHOLD) | (wind > BAD_WEATHER_WIND_THRESHOLD)).astype(float)
    return out


def _apply_weather_enrichment(df, use_weather_api=False, enrich_only_unobserved=False):
    out = df.copy()
    non_derived_weather_cols = ["weather_temp_mean_c", "weather_precipitation_mm", "weather_rain_mm", "weather_windspeed_max_kmh"]
    for col in WEATHER_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    existing_weather_mask = out[[c for c in non_derived_weather_cols if c in out.columns]].notna().any(axis=1)
    target_mask = pd.Series(True, index=out.index)
    if enrich_only_unobserved and "is_observed" in out.columns:
        target_mask = ~_to_bool_series(out["is_observed"])

    weather_info = {
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
        weather_info.update(api_stats)

        if len(weather_df) > 0:
            out = out.merge(weather_df, on="match_id", how="left", suffixes=("", "_api"))
            for col in WEATHER_COLUMNS:
                api_col = f"{col}_api"
                if api_col in out.columns:
                    missing_mask = out[col].isna() & target_mask
                    api_values = _safe_numeric(out[api_col])
                    filled_now = int((missing_mask & api_values.notna()).sum())
                    weather_info["filled_cells"] += filled_now
                    out.loc[missing_mask, col] = api_values.loc[missing_mask]
                    out = out.drop(columns=[api_col])

            if "weather_source" in out.columns and "weather_source_api" in out.columns:
                out["weather_source"] = out["weather_source"].fillna(out["weather_source_api"])
                out = out.drop(columns=["weather_source_api"])
            elif "weather_source_api" in out.columns:
                out["weather_source"] = out["weather_source_api"]
                out = out.drop(columns=["weather_source_api"])

    if "weather_source" not in out.columns:
        out["weather_source"] = ""
    out["weather_source"] = out["weather_source"].astype(str)

    out.loc[existing_weather_mask, "weather_source"] = "source_table"
    out = _refresh_weather_bad_flag(out)

    enriched_mask = out[[c for c in non_derived_weather_cols if c in out.columns]].notna().any(axis=1)
    if target_mask.any():
        weather_info["rows_enriched"] = int((enriched_mask & target_mask).sum())
    return out, weather_info


def _build_feature_table(match_df, context_df, tickets_df, trends_df, articles_df):
    merged = match_df.merge(context_df, on="match_id", how="left")
    merged = merged.merge(tickets_df, on="match_id", how="left")
    merged = merged.merge(articles_df, on="match_id", how="left")
    merged = merged.merge(_compute_interest_feature(match_df, trends_df), on="match_id", how="left")
    merged = _add_calendar_features(merged)
    merged = _infer_matchday(merged)
    merged = _normalize_types(merged)
    merged = merged.sort_values([DATE_COLUMN, "kickoff_hour", "match_id"]).reset_index(drop=True)
    lag_df = _compute_lag_features(merged)
    merged = merged.merge(lag_df, on="match_id", how="left")
    merged = _normalize_types(merged)
    return merged


def _apply_inference_fallbacks(df):
    out = df.copy()
    observed_mask = _to_bool_series(out["is_observed"]) & _to_bool_series(out["is_home_match"]) & out[TARGET_COLUMN].notna()
    observed_vals = _safe_numeric(out.loc[observed_mask, TARGET_COLUMN])
    global_mean = float(observed_vals.mean()) if observed_vals.notna().any() else 0.0
    inference_mask = ~_to_bool_series(out["is_observed"])

    fallback_counts = {}
    for col in ["attendance_last_match", "attendance_last_3_avg"]:
        missing_mask = inference_mask & out[col].isna()
        fallback_counts[col] = int(missing_mask.sum())
        out.loc[missing_mask, col] = global_mean

    return out, {"global_mean": global_mean, "fallback_counts": fallback_counts}


def build_match_level_dataset(tables, use_weather_api=False, return_stats=False):
    match_df = _prepare_match(tables["match"])
    context_df = _prepare_context(tables.get("context", pd.DataFrame()))
    tickets_df = _prepare_tickets(tables.get("tickets", pd.DataFrame()))
    trends_df = _prepare_trends(tables.get("trends", pd.DataFrame()))
    articles_df = _prepare_articles(tables.get("articles", pd.DataFrame()))

    match_df["is_observed"] = True
    merged = _build_feature_table(match_df, context_df, tickets_df, trends_df, articles_df)
    merged, weather_stats = _apply_weather_enrichment(merged, use_weather_api=use_weather_api, enrich_only_unobserved=False)
    merged = merged[merged["is_home_match"]].copy()
    merged = merged.dropna(subset=[TARGET_COLUMN]).copy()
    merged = merged.sort_values([DATE_COLUMN, "kickoff_hour", "match_id"]).reset_index(drop=True)
    if return_stats:
        return merged, {"weather": weather_stats}
    return merged


def build_inference_dataset(tables, new_matches_df, return_stats=False, use_weather_api=False):
    historical_match_df = _prepare_match(tables["match"])
    context_df = _prepare_context(tables.get("context", pd.DataFrame()))
    tickets_df = _prepare_tickets(tables.get("tickets", pd.DataFrame()))
    trends_df = _prepare_trends(tables.get("trends", pd.DataFrame()))
    articles_df = _prepare_articles(tables.get("articles", pd.DataFrame()))

    historical_match_df["is_observed"] = True

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

    union_cols = sorted(set(historical_match_df.columns).union(set(incoming.columns)))
    historical_aligned = historical_match_df.reindex(columns=union_cols)
    incoming_aligned = incoming.reindex(columns=union_cols)
    combined = pd.concat([historical_aligned, incoming_aligned], ignore_index=True)

    merged = _build_feature_table(combined, context_df, tickets_df, trends_df, articles_df)
    merged, weather_stats = _apply_weather_enrichment(merged, use_weather_api=use_weather_api, enrich_only_unobserved=True)
    merged, fallback_stats = _apply_inference_fallbacks(merged)
    inference_only = merged[~_to_bool_series(merged["is_observed"])].copy()
    inference_only = inference_only.sort_values([DATE_COLUMN, "kickoff_hour", "match_id"]).reset_index(drop=True)

    if return_stats:
        stats = {
            "user_input_features": USER_INPUT_FEATURES,
            "auto_generated_features": AUTO_GENERATED_FEATURES,
            "fallback": fallback_stats,
            "weather": weather_stats,
        }
        return inference_only, stats
    return inference_only


def get_feature_columns(df):
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def get_full_reference_feature_columns(df):
    return [col for col in FULL_REFERENCE_COLUMNS if col in df.columns]


def get_feature_minimization_groups(df):
    available_columns = set(df.columns)
    groups = {}
    for group_name, columns in FEATURE_MINIMIZATION_GROUPS.items():
        selected = [col for col in columns if col in available_columns]
        if len(selected) > 0:
            groups[group_name] = selected
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
            if numeric.notna().any():
                fill_values[col] = float(numeric.median())
            else:
                fill_values[col] = 0.0
        else:
            non_null = series.dropna().astype(str)
            if len(non_null) == 0:
                fill_values[col] = "unknown"
            else:
                fill_values[col] = str(non_null.mode().iloc[0])
    return fill_values


def prepare_inference_features(df, feature_columns, fill_values=None):
    out = _normalize_types(df.copy())
    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan

    for col in feature_columns:
        series = out[col]
        if pd.api.types.is_numeric_dtype(series):
            fallback = 0.0
            if isinstance(fill_values, dict) and col in fill_values:
                fallback = float(fill_values[col])
            numeric = _safe_numeric(series)
            if not np.isfinite(fallback):
                fallback = float(numeric.mean()) if numeric.notna().any() else 0.0
            out[col] = numeric.fillna(fallback)
        else:
            fallback = "unknown"
            if isinstance(fill_values, dict) and col in fill_values:
                fallback = str(fill_values[col])
            out[col] = series.astype(str).replace({"nan": "unknown", "None": "unknown"}).fillna(fallback)

    return out[feature_columns].copy()
