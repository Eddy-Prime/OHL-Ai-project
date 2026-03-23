import numpy as np
import pandas as pd

from .config import ADDITIONAL_FEATURE_COLUMNS, DATE_COLUMN, REQUIRED_FEATURE_COLUMNS, TARGET_COLUMN


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
    df["is_home_match"] = _to_bool_series(df["is_home_match"])
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    for col in ["matchday", "goals_home_ft", "goals_away_ft"]:
        if col in df.columns:
            df[col] = _safe_numeric(df[col])

    df = df.dropna(subset=[DATE_COLUMN])
    df = df.drop_duplicates(subset="match_id")

    kickoff = pd.to_datetime(df["kickoff_time_local"], format="%H:%M:%S", errors="coerce")
    df["kickoff_hour"] = kickoff.dt.hour.fillna(0).astype(int)
    df["month"] = df[DATE_COLUMN].dt.month.astype(int)

    df["opponent"] = np.where(df["is_home_match"], df.get("away_team"), df.get("home_team"))
    df["ohl_goals"] = np.where(df["is_home_match"], df.get("goals_home_ft"), df.get("goals_away_ft"))
    df["opp_goals"] = np.where(df["is_home_match"], df.get("goals_away_ft"), df.get("goals_home_ft"))
    df["ohl_goals"] = _safe_numeric(df["ohl_goals"])
    df["opp_goals"] = _safe_numeric(df["opp_goals"])

    df["points"] = np.where(df["ohl_goals"] > df["opp_goals"], 3, np.where(df["ohl_goals"] == df["opp_goals"], 1, 0))
    missing_goals = df["ohl_goals"].isna() | df["opp_goals"].isna()
    df.loc[missing_goals, "points"] = np.nan
    df["is_win"] = np.where(df["ohl_goals"] > df["opp_goals"], 1, 0)
    df.loc[missing_goals, "is_win"] = np.nan

    df = df.sort_values([DATE_COLUMN, "kickoff_hour", "match_id"]).reset_index(drop=True)
    return df


def _prepare_context(df_context):
    df = df_context.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
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
    numeric_cols = [
        "weather_temp_mean_c",
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
        "promo_tickets_total",
        "pct_free_tickets",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].drop_duplicates(subset="match_id")
    return df


def _prepare_tickets(df_tickets):
    df = df_tickets.copy()
    for col in ["seasonpass_holders"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep_cols = ["match_id", "seasonpass_holders"]
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].drop_duplicates(subset="match_id")
    return df


def _prepare_trends(df_trends):
    df = df_trends.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ohl_interest"] = pd.to_numeric(df["ohl_interest"], errors="coerce")
    df = df.dropna(subset=["date", "ohl_interest"]).copy()
    trends = df.groupby("date", as_index=False)["ohl_interest"].mean()
    trends = trends.sort_values("date").reset_index(drop=True)
    trends["ohl_interest_3d_avg"] = trends["ohl_interest"].rolling(window=3, min_periods=1).mean()
    trends["ohl_interest_7d_avg"] = trends["ohl_interest"].rolling(window=7, min_periods=1).mean()
    trends["ohl_interest_trend"] = trends["ohl_interest"].rolling(window=3, min_periods=1).apply(
        lambda x: _compute_trend_slope(x.values, window=min(3, len(x))),
        raw=False
    )
    return trends


def _prepare_articles(df_articles):
    df = df_articles.copy()
    df = df[df["match_id"].notna()].copy()
    df["days_to_match"] = pd.to_numeric(df["days_to_match"], errors="coerce")
    if "article_id" not in df.columns:
        df["article_id"] = np.arange(len(df))
    df["is_article_1d"] = ((df["days_to_match"] >= 0) & (df["days_to_match"] <= 1)).astype(int)
    df["is_article_3d"] = ((df["days_to_match"] >= 0) & (df["days_to_match"] <= 3)).astype(int)
    df["is_article_7d"] = ((df["days_to_match"] >= 0) & (df["days_to_match"] <= 7)).astype(int)
    grouped = df.groupby("match_id", as_index=False).agg(
        num_articles=("article_id", "count"),
        avg_days_to_match=("days_to_match", "mean"),
        num_articles_1d=("is_article_1d", "sum"),
        num_articles_3d=("is_article_3d", "sum"),
        num_articles_7d=("is_article_7d", "sum"),
    )
    return grouped


def _compute_trend_features(match_df, trends_df):
    out = match_df[["match_id", DATE_COLUMN]].copy()
    out["trend_lookup_date"] = out[DATE_COLUMN].dt.normalize() - pd.Timedelta(days=1)
    merged = pd.merge_asof(
        out.sort_values("trend_lookup_date"),
        trends_df.sort_values("date"),
        left_on="trend_lookup_date",
        right_on="date",
        direction="backward",
    )
    merged = merged[["match_id", "ohl_interest", "ohl_interest_3d_avg", "ohl_interest_7d_avg", "ohl_interest_trend"]]
    merged = merged.rename(columns={"ohl_interest": "ohl_interest_last_available"})
    merged["ohl_interest"] = merged["ohl_interest_last_available"]
    return merged


def _window_mean(values, window):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values[-window:]))


def _window_sum(values, window):
    if len(values) == 0:
        return 0.0
    return float(np.sum(values[-window:]))


def _window_std(values, window):
    if len(values) < window:
        return np.nan
    return float(np.std(values[-window:], ddof=0))


def _compute_ewm(values, span=3):
    if len(values) < 2:
        return np.nan
    values_array = np.asarray(values[-span*2:], dtype=float)
    if len(values_array) == 0 or np.all(np.isnan(values_array)):
        return np.nan
    try:
        series = pd.Series(values_array)
        ewm_result = series.ewm(span=span, adjust=False).mean()
        return float(ewm_result.iloc[-1])
    except:
        return np.nan


def _compute_trend_slope(values, window=3):
    if len(values) < window:
        return np.nan
    recent = np.asarray(values[-window:], dtype=float)
    if len(recent) < 2 or np.all(np.isnan(recent)):
        return np.nan
    valid_indices = np.where(~np.isnan(recent))[0]
    if len(valid_indices) < 2:
        return np.nan
    try:
        x = valid_indices.astype(float)
        y = recent[valid_indices]
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except:
        return np.nan


def _count_unbeaten(results):
    count = 0
    for result in reversed(results):
        if np.isnan(result):
            continue
        if result > 0 or result == 1:
            count += 1
        else:
            break
    return float(count)



def _compute_history_features(df):
    home_attendance_history = []
    attendance_by_weekday = {}
    attendance_vs_opponent = {}
    points_history = []
    wins_history = []
    goals_for_history = []
    goals_against_history = []
    opponent_stats = {}
    opponent_goals_for = {}
    opponent_goals_against = {}
    home_match_count = 0
    last_home_date = None
    last_match_date = None

    rows = []
    for _, row in df.iterrows():
        match_date = row[DATE_COLUMN]
        opponent = str(row.get("opponent", "unknown"))
        weekday = str(row.get("weekday_name", "unknown")).lower()

        attendance_last_match = home_attendance_history[-1] if len(home_attendance_history) >= 1 else np.nan
        attendance_last_3_avg = _window_mean(home_attendance_history, 3)
        attendance_last_5_avg = _window_mean(home_attendance_history, 5)
        attendance_rolling_std_3 = _window_std(home_attendance_history, 3)
        attendance_std_last_5 = _window_std(home_attendance_history, 5)
        attendance_ewm = _compute_ewm(home_attendance_history, span=3)
        
        weekday_history = attendance_by_weekday.get(weekday, [])
        attendance_last_same_weekday = weekday_history[-1] if len(weekday_history) >= 1 else np.nan
        
        opponent_history = attendance_vs_opponent.get(opponent, [])
        attendance_vs_same_opponent_last = opponent_history[-1] if len(opponent_history) >= 1 else np.nan

        points_last_1 = points_history[-1] if len(points_history) >= 1 else np.nan
        points_last_3_matches = _window_sum(points_history, 3)
        points_last_5_matches = _window_sum(points_history, 5)
        
        wins_last_1 = wins_history[-1] if len(wins_history) >= 1 else np.nan
        wins_last_3_matches = _window_sum(wins_history, 3)
        wins_last_5 = _window_sum(wins_history, 5)
        unbeaten_streak = _count_unbeaten(wins_history)
        
        goals_scored_last_1 = goals_for_history[-1] if len(goals_for_history) >= 1 else np.nan
        goals_scored_last_3_matches = _window_sum(goals_for_history, 3)
        goals_scored_last_5 = _window_sum(goals_for_history, 5)
        
        goals_conceded_last_1 = goals_against_history[-1] if len(goals_against_history) >= 1 else np.nan
        goals_conceded_last_3_matches = _window_sum(goals_against_history, 3)
        goals_conceded_last_5 = _window_sum(goals_against_history, 5)
        
        goal_difference_last_1 = goals_scored_last_1 - goals_conceded_last_1 if pd.notna(goals_scored_last_1) and pd.notna(goals_conceded_last_1) else np.nan
        goal_difference_last_3_matches = goals_scored_last_3_matches - goals_conceded_last_3_matches
        goal_difference_last_5 = goals_scored_last_5 - goals_conceded_last_5

        current_opponent_stats = opponent_stats.get(opponent, {"count": 0, "sum_attendance": 0.0})
        if current_opponent_stats["count"] > 0:
            opponent_historical_avg_attendance = current_opponent_stats["sum_attendance"] / current_opponent_stats["count"]
        else:
            opponent_historical_avg_attendance = np.nan
        opponent_frequency_seen = float(current_opponent_stats["count"])
        
        opponent_avg_gf = opponent_goals_for.get(opponent, [])
        opponent_avg_goals_scored = float(np.mean(opponent_avg_gf)) if len(opponent_avg_gf) > 0 else np.nan
        opponent_recent_goals_scored = _window_mean(opponent_avg_gf, 3) if len(opponent_avg_gf) > 0 else np.nan
        
        opponent_avg_ga = opponent_goals_against.get(opponent, [])
        opponent_avg_goals_conceded = float(np.mean(opponent_avg_ga)) if len(opponent_avg_ga) > 0 else np.nan
        opponent_recent_goals_conceded = _window_mean(opponent_avg_ga, 3) if len(opponent_avg_ga) > 0 else np.nan

        means = []
        for stats in opponent_stats.values():
            if stats["count"] > 0:
                means.append(stats["sum_attendance"] / stats["count"])
        if len(means) > 0 and not np.isnan(opponent_historical_avg_attendance):
            rank_position = float(np.sum(np.asarray(means) <= opponent_historical_avg_attendance))
            opponent_encoded_rank_proxy = rank_position / float(len(means))
            opponent_strength_proxy = opponent_encoded_rank_proxy
            big_threshold = float(np.quantile(means, 0.75))
            big_opponent_flag = float(opponent_historical_avg_attendance >= big_threshold and current_opponent_stats["count"] >= 2)
        else:
            opponent_encoded_rank_proxy = np.nan
            opponent_strength_proxy = np.nan
            big_opponent_flag = 0.0

        days_since_previous_home_match = np.nan if last_home_date is None else float((match_date - last_home_date).days)
        days_since_previous_match = np.nan if last_match_date is None else float((match_date - last_match_date).days)
        consecutive_home_matches = float(home_match_count) if bool(row.get("is_home_match", False)) else np.nan

        matchday_value = row.get("matchday")
        month_value = row.get("month")
        early_season_flag = float((pd.notna(matchday_value) and matchday_value <= 6) or (pd.notna(month_value) and int(month_value) in [7, 8]))
        mid_season_flag = float((pd.notna(matchday_value) and 7 <= matchday_value <= 23) or (pd.notna(month_value) and int(month_value) in [9, 10, 11, 12, 1, 2]))
        late_season_flag = float((pd.notna(matchday_value) and matchday_value >= 24) or (pd.notna(month_value) and int(month_value) in [3, 4, 5]))

        rows.append(
            {
                "match_id": row["match_id"],
                "attendance_last_match": attendance_last_match,
                "attendance_last_3_avg": attendance_last_3_avg,
                "attendance_last_5_avg": attendance_last_5_avg,
                "attendance_rolling_std_3": attendance_rolling_std_3,
                "attendance_std_last_5": attendance_std_last_5,
                "attendance_ewm": attendance_ewm,
                "attendance_last_same_weekday": attendance_last_same_weekday,
                "attendance_vs_same_opponent_last": attendance_vs_same_opponent_last,
                "points_last_1": points_last_1,
                "points_last_3_matches": points_last_3_matches,
                "points_last_5_matches": points_last_5_matches,
                "wins_last_1": wins_last_1,
                "wins_last_3_matches": wins_last_3_matches,
                "wins_last_5": wins_last_5,
                "unbeaten_streak": unbeaten_streak,
                "goals_scored_last_1": goals_scored_last_1,
                "goals_scored_last_3_matches": goals_scored_last_3_matches,
                "goals_scored_last_5": goals_scored_last_5,
                "goals_conceded_last_1": goals_conceded_last_1,
                "goals_conceded_last_3_matches": goals_conceded_last_3_matches,
                "goals_conceded_last_5": goals_conceded_last_5,
                "goal_difference_last_1": goal_difference_last_1,
                "goal_difference_last_3_matches": goal_difference_last_3_matches,
                "goal_difference_last_5": goal_difference_last_5,
                "opponent_historical_avg_attendance": opponent_historical_avg_attendance,
                "opponent_frequency_seen": opponent_frequency_seen,
                "opponent_avg_goals_scored": opponent_avg_goals_scored,
                "opponent_avg_goals_conceded": opponent_avg_goals_conceded,
                "opponent_strength_proxy": opponent_strength_proxy,
                "opponent_recent_goals_scored": opponent_recent_goals_scored,
                "opponent_recent_goals_conceded": opponent_recent_goals_conceded,
                "big_opponent_flag": big_opponent_flag,
                "opponent_encoded_rank_proxy": opponent_encoded_rank_proxy,
                "days_since_previous_home_match": days_since_previous_home_match,
                "days_since_previous_match": days_since_previous_match,
                "consecutive_home_matches": consecutive_home_matches,
                "early_season_flag": early_season_flag,
                "mid_season_flag": mid_season_flag,
                "late_season_flag": late_season_flag,
                "ohl_goals_last_3_matches": goals_scored_last_3_matches,
                "opp_goals_last_3_matches": goals_conceded_last_3_matches,
            }
        )

        is_observed = bool(row.get("is_observed", True))
        if is_observed:
            if pd.notna(row.get("points")):
                points_history.append(float(row["points"]))
            if pd.notna(row.get("is_win")):
                wins_history.append(float(row["is_win"]))
            if pd.notna(row.get("ohl_goals")):
                goals_for_history.append(float(row["ohl_goals"]))
            if pd.notna(row.get("opp_goals")):
                goals_against_history.append(float(row["opp_goals"]))

            if bool(row.get("is_home_match", False)) and pd.notna(row.get(TARGET_COLUMN)):
                attendance_value = float(row[TARGET_COLUMN])
                home_attendance_history.append(attendance_value)
                
                if weekday not in attendance_by_weekday:
                    attendance_by_weekday[weekday] = []
                attendance_by_weekday[weekday].append(attendance_value)
                
                if opponent not in attendance_vs_opponent:
                    attendance_vs_opponent[opponent] = []
                attendance_vs_opponent[opponent].append(attendance_value)
                
                current = opponent_stats.get(opponent, {"count": 0, "sum_attendance": 0.0})
                current["count"] += 1
                current["sum_attendance"] += attendance_value
                opponent_stats[opponent] = current
                
                last_home_date = match_date
                home_match_count += 1

            if pd.notna(row.get("ohl_goals")):
                if opponent not in opponent_goals_for:
                    opponent_goals_for[opponent] = []
                opponent_goals_for[opponent].append(float(row["ohl_goals"]))
                
            if pd.notna(row.get("opp_goals")):
                if opponent not in opponent_goals_against:
                    opponent_goals_against[opponent] = []
                opponent_goals_against[opponent].append(float(row["opp_goals"]))

            last_match_date = match_date

    return pd.DataFrame(rows)


def _normalize_types(df):
    out = df.copy()
    for col in ["season", "stage", "competition_name", "away_team", "weekday_name"]:
        if col in out.columns:
            out[col] = out[col].astype(str)

    bool_cols = ["is_weekend", "is_midweek", "is_public_holiday", "is_school_holiday_flanders", "has_promotion"]
    for col in bool_cols:
        if col in out.columns:
            out[col] = _to_bool_series(out[col]).astype(float)

    numeric_cols = [
        "matchday",
        "kickoff_hour",
        "month",
        "weather_temp_mean_c",
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
        "promo_tickets_total",
        "pct_free_tickets",
        "seasonpass_holders",
        "num_articles",
        "avg_days_to_match",
        "num_articles_1d",
        "num_articles_3d",
        "num_articles_7d",
        "articles_trend_slope",
        "ohl_interest",
        "ohl_interest_last_available",
        "ohl_interest_3d_avg",
        "ohl_interest_7d_avg",
        "ohl_interest_trend",
        "attendance_last_match",
        "attendance_last_3_avg",
        "attendance_last_5_avg",
        "attendance_rolling_std_3",
        "attendance_std_last_5",
        "attendance_ewm",
        "attendance_last_same_weekday",
        "attendance_vs_same_opponent_last",
        "points_last_1",
        "points_last_3_matches",
        "points_last_5_matches",
        "wins_last_1",
        "wins_last_3_matches",
        "wins_last_5",
        "unbeaten_streak",
        "goals_scored_last_1",
        "goals_scored_last_3_matches",
        "goals_scored_last_5",
        "goals_conceded_last_1",
        "goals_conceded_last_3_matches",
        "goals_conceded_last_5",
        "goal_difference_last_1",
        "goal_difference_last_3_matches",
        "goal_difference_last_5",
        "opponent_historical_avg_attendance",
        "opponent_frequency_seen",
        "opponent_avg_goals_scored",
        "opponent_avg_goals_conceded",
        "opponent_strength_proxy",
        "opponent_recent_goals_scored",
        "opponent_recent_goals_conceded",
        "big_opponent_flag",
        "opponent_encoded_rank_proxy",
        "days_since_previous_home_match",
        "days_since_previous_match",
        "consecutive_home_matches",
        "early_season_flag",
        "mid_season_flag",
        "late_season_flag",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = _safe_numeric(out[col])

    has_promotion = out["has_promotion"] if "has_promotion" in out.columns else pd.Series(0.0, index=out.index)
    is_weekend = out["is_weekend"] if "is_weekend" in out.columns else pd.Series(0.0, index=out.index)
    big_opponent_flag = out["big_opponent_flag"] if "big_opponent_flag" in out.columns else pd.Series(0.0, index=out.index)
    pct_free_tickets = out["pct_free_tickets"] if "pct_free_tickets" in out.columns else pd.Series(0.0, index=out.index)
    seasonpass_holders = out["seasonpass_holders"] if "seasonpass_holders" in out.columns else pd.Series(0.0, index=out.index)
    ohl_interest_val = out["ohl_interest"] if "ohl_interest" in out.columns else pd.Series(0.0, index=out.index)

    out["promotion_weekend_interaction"] = _safe_numeric(has_promotion).fillna(0) * _safe_numeric(is_weekend).fillna(0)
    out["promotion_big_opponent_interaction"] = _safe_numeric(has_promotion).fillna(0) * _safe_numeric(big_opponent_flag).fillna(0)
    out["weekend_big_opponent_interaction"] = _safe_numeric(is_weekend).fillna(0) * _safe_numeric(big_opponent_flag).fillna(0)
    out["ohl_interest_big_opponent_interaction"] = _safe_numeric(ohl_interest_val).fillna(0) * _safe_numeric(big_opponent_flag).fillna(0)
    out["free_ticket_pressure"] = _safe_numeric(pct_free_tickets).fillna(0) * _safe_numeric(seasonpass_holders).fillna(0)
    
    return out



def _build_feature_table(match_df, context_df, tickets_df, trends_df, articles_df):
    merged = match_df.merge(context_df, on="match_id", how="left")
    merged = merged.merge(tickets_df, on="match_id", how="left")
    merged = merged.merge(articles_df, on="match_id", how="left")
    trend_features = _compute_trend_features(match_df, trends_df)
    merged = merged.merge(trend_features, on="match_id", how="left")

    merged = _normalize_types(merged)
    merged = merged.sort_values([DATE_COLUMN, "kickoff_hour", "match_id"]).reset_index(drop=True)
    history_features = _compute_history_features(merged)
    merged = merged.merge(history_features, on="match_id", how="left")
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
        if "opponent" in incoming.columns:
            incoming["away_team"] = incoming["opponent"]
        else:
            incoming["away_team"] = "unknown"

    incoming["opponent"] = incoming["away_team"].astype(str)
    incoming[TARGET_COLUMN] = np.nan
    incoming["ohl_goals"] = np.nan
    incoming["opp_goals"] = np.nan
    incoming["points"] = np.nan
    incoming["is_win"] = np.nan
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
    columns = REQUIRED_FEATURE_COLUMNS + ADDITIONAL_FEATURE_COLUMNS
    return [col for col in columns if col in df.columns]


def split_features_target(df, feature_columns):
    x = df[feature_columns].copy()
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
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
    out = df.copy()

    out = _normalize_types(out)

    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan

    return out[feature_columns].copy()


