import numpy as np
import pandas as pd

from .config import DATE_COLUMN, TARGET_COLUMN


FEATURE_COLUMNS = [
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
    "weather_temp_mean_c",
    "weather_rain_mm",
    "weather_windspeed_max_kmh",
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


def _to_bool_series(series):
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y", "t"])


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _prepare_match(df_match):
    df = df_match.copy()
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

    kickoff = pd.to_datetime(df.get("kickoff_time_local"), format="%H:%M:%S", errors="coerce")
    df["kickoff_hour"] = kickoff.dt.hour.fillna(0).astype(int)
    df["month"] = df[DATE_COLUMN].dt.month.astype(int)

    if "away_team" not in df.columns:
        df["away_team"] = "unknown"

    df = df.sort_values([DATE_COLUMN, "kickoff_hour", "match_id"]).reset_index(drop=True)
    return df


def _prepare_context(df_context):
    df = df_context.copy()
    bool_cols = [
        "has_promotion",
        "is_weekend",
        "is_midweek",
        "is_public_holiday",
        "is_school_holiday_flanders",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = _to_bool_series(df[col]).fillna(False)

    numeric_cols = ["weather_temp_mean_c", "weather_rain_mm", "weather_windspeed_max_kmh", "promo_tickets_total", "pct_free_tickets"]
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
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
        "has_promotion",
        "promo_tickets_total",
        "pct_free_tickets",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].drop_duplicates(subset="match_id")


def _prepare_tickets(df_tickets):
    df = df_tickets.copy()
    if "seasonpass_holders" in df.columns:
        df["seasonpass_holders"] = _safe_numeric(df["seasonpass_holders"])
    keep_cols = [c for c in ["match_id", "seasonpass_holders"] if c in df.columns]
    return df[keep_cols].drop_duplicates(subset="match_id")


def _prepare_trends(df_trends):
    df = df_trends.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ohl_interest"] = _safe_numeric(df["ohl_interest"])
    df = df.dropna(subset=["date", "ohl_interest"]).copy()
    trend = df.groupby("date", as_index=False)["ohl_interest"].mean()
    trend = trend.sort_values("date").reset_index(drop=True)
    trend["ohl_interest_3d_avg"] = trend["ohl_interest"].rolling(window=3, min_periods=1).mean()
    return trend


def _prepare_articles(df_articles):
    df = df_articles.copy()
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
    rows = []
    for _, row in df.iterrows():
        last_match = history[-1] if len(history) >= 1 else np.nan
        last_3_avg = float(np.mean(history[-3:])) if len(history) >= 1 else np.nan
        rows.append(
            {
                "match_id": row["match_id"],
                "attendance_last_match": last_match,
                "attendance_last_3_avg": last_3_avg,
            }
        )

        if bool(row.get("is_observed", True)) and bool(row.get("is_home_match", False)) and pd.notna(row.get(TARGET_COLUMN)):
            history.append(float(row[TARGET_COLUMN]))

    return pd.DataFrame(rows)


def _normalize_types(df):
    out = df.copy()

    for col in ["season", "stage", "competition_name", "away_team", "weekday_name"]:
        if col in out.columns:
            out[col] = out[col].astype(str)

    for col in ["is_weekend", "is_midweek", "is_public_holiday", "is_school_holiday_flanders", "has_promotion"]:
        if col in out.columns:
            out[col] = _to_bool_series(out[col]).astype(float)

    for col in [
        "matchday",
        "kickoff_hour",
        "month",
        "weather_temp_mean_c",
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
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

    return out


def _build_feature_table(match_df, context_df, tickets_df, trends_df, articles_df):
    merged = match_df.merge(context_df, on="match_id", how="left")
    merged = merged.merge(tickets_df, on="match_id", how="left")
    merged = merged.merge(articles_df, on="match_id", how="left")
    merged = merged.merge(_compute_interest_feature(match_df, trends_df), on="match_id", how="left")
    merged = _normalize_types(merged)
    merged = merged.sort_values([DATE_COLUMN, "kickoff_hour", "match_id"]).reset_index(drop=True)
    lag_df = _compute_lag_features(merged)
    merged = merged.merge(lag_df, on="match_id", how="left")
    merged = _normalize_types(merged)
    return merged


def build_match_level_dataset(tables):
    match_df = _prepare_match(tables["match"])
    context_df = _prepare_context(tables["context"])
    tickets_df = _prepare_tickets(tables["tickets"])
    trends_df = _prepare_trends(tables["trends"])
    articles_df = _prepare_articles(tables["articles"])

    match_df["is_observed"] = True
    merged = _build_feature_table(match_df, context_df, tickets_df, trends_df, articles_df)
    merged = merged[merged["is_home_match"]].copy()
    merged = merged.dropna(subset=[TARGET_COLUMN]).copy()
    merged = merged.sort_values(DATE_COLUMN).reset_index(drop=True)
    return merged


def build_inference_dataset(tables, new_matches_df):
    historical_match_df = _prepare_match(tables["match"])
    context_df = _prepare_context(tables["context"])
    tickets_df = _prepare_tickets(tables["tickets"])
    trends_df = _prepare_trends(tables["trends"])
    articles_df = _prepare_articles(tables["articles"])

    historical_match_df["is_observed"] = True

    incoming = new_matches_df.copy()
    if DATE_COLUMN not in incoming.columns:
        raise ValueError(f"Missing required column in input file: {DATE_COLUMN}")
    incoming[DATE_COLUMN] = pd.to_datetime(incoming[DATE_COLUMN], errors="coerce")
    if incoming[DATE_COLUMN].isna().any():
        raise ValueError("Some rows in input file have invalid match_date values")

    if "match_id" not in incoming.columns:
        incoming["match_id"] = [f"new_match_{i}" for i in range(len(incoming))]
    incoming["match_id"] = incoming["match_id"].astype(str)

    if "is_home_match" not in incoming.columns:
        incoming["is_home_match"] = True
    incoming["is_home_match"] = _to_bool_series(incoming["is_home_match"])

    if "kickoff_hour" not in incoming.columns:
        kickoff = pd.to_datetime(incoming.get("kickoff_time_local"), format="%H:%M:%S", errors="coerce")
        incoming["kickoff_hour"] = kickoff.dt.hour.fillna(0).astype(int)
    if "month" not in incoming.columns:
        incoming["month"] = incoming[DATE_COLUMN].dt.month
    if "away_team" not in incoming.columns:
        incoming["away_team"] = incoming.get("opponent", "unknown")

    incoming[TARGET_COLUMN] = np.nan
    incoming["is_observed"] = False

    union_cols = sorted(set(historical_match_df.columns).union(set(incoming.columns)))
    historical_aligned = historical_match_df.reindex(columns=union_cols)
    incoming_aligned = incoming.reindex(columns=union_cols)
    combined = pd.concat([historical_aligned, incoming_aligned], ignore_index=True)

    merged = _build_feature_table(combined, context_df, tickets_df, trends_df, articles_df)
    merged = merged[~merged["is_observed"]].copy()
    merged = merged.sort_values(DATE_COLUMN).reset_index(drop=True)
    return merged


def get_feature_columns(df):
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def split_features_target(df, feature_columns):
    x = df[feature_columns].copy()
    y = _safe_numeric(df[TARGET_COLUMN])
    valid = y.notna()
    x = x.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    meta = df.loc[valid, ["match_id", DATE_COLUMN, "away_team"]].reset_index(drop=True)
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


def prepare_inference_features(df, feature_columns):
    out = _normalize_types(df.copy())
    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[feature_columns].copy()
