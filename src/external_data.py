from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _normalize_team_name(series):
    return series.fillna("unknown").astype(str).str.strip().str.lower()


def _points_from_goal_diff(goal_diff):
    return np.where(goal_diff > 0, 3.0, np.where(goal_diff == 0, 1.0, 0.0))


def load_transfermarkt_matches(external_csv_path):
    path = Path(external_csv_path)
    if not path.exists():
        return pd.DataFrame(), {"enabled": False, "exists": False, "path": str(path), "rows_loaded": 0}

    try:
        raw = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame(), {"enabled": True, "exists": True, "path": str(path), "rows_loaded": 0}

    if len(raw) == 0:
        return pd.DataFrame(), {"enabled": True, "exists": True, "path": str(path), "rows_loaded": 0}

    df = raw.copy()
    if "match_date" not in df.columns:
        return pd.DataFrame(), {"enabled": True, "exists": True, "path": str(path), "rows_loaded": int(len(raw))}

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if "away_team" not in df.columns:
        if "opponent_name_normalized" in df.columns:
            df["away_team"] = df["opponent_name_normalized"]
        elif "opponent_name_raw" in df.columns:
            df["away_team"] = df["opponent_name_raw"]
        else:
            df["away_team"] = "unknown"

    if "attendance" not in df.columns:
        df["attendance"] = np.nan
    df["attendance"] = _safe_numeric(df["attendance"])

    if "goals_for" not in df.columns:
        df["goals_for"] = np.nan
    if "goals_against" not in df.columns:
        df["goals_against"] = np.nan
    df["goals_for"] = _safe_numeric(df["goals_for"])
    df["goals_against"] = _safe_numeric(df["goals_against"])

    if "is_home_match" in df.columns:
        mask_home = df["is_home_match"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])
        df = df[mask_home].copy()

    df = df.dropna(subset=["match_date"]).copy()
    df["away_team_norm"] = _normalize_team_name(df["away_team"])
    df = df.sort_values("match_date").reset_index(drop=True)

    stats = {
        "enabled": True,
        "exists": True,
        "path": str(path),
        "rows_loaded": int(len(raw)),
        "rows_home_rows": int(len(df)),
        "attendance_rows": int(df["attendance"].notna().sum()),
    }
    return df, stats


def _build_priors_from_base(base, min_matches):
    global_attendance_median = float(base["attendance"].median()) if base["attendance"].notna().any() else np.nan
    grouped = base.groupby("away_team_norm", as_index=False).agg(
        tm_opponent_attendance_median=("attendance", "median"),
        tm_opponent_goal_diff_avg=("goal_diff", "mean"),
        tm_opponent_points_avg=("points", "mean"),
        tm_opponent_match_count=("away_team_norm", "count"),
    )
    grouped = grouped[grouped["tm_opponent_match_count"] >= int(min_matches)].copy()
    if len(grouped) == 0:
        grouped = base.groupby("away_team_norm", as_index=False).agg(
            tm_opponent_attendance_median=("attendance", "median"),
            tm_opponent_goal_diff_avg=("goal_diff", "mean"),
            tm_opponent_points_avg=("points", "mean"),
            tm_opponent_match_count=("away_team_norm", "count"),
        )

    if np.isfinite(global_attendance_median) and global_attendance_median > 0:
        grouped["tm_opponent_attendance_index"] = grouped["tm_opponent_attendance_median"] / global_attendance_median
    else:
        grouped["tm_opponent_attendance_index"] = np.nan

    q_low = float(grouped["tm_opponent_attendance_index"].quantile(0.33)) if grouped["tm_opponent_attendance_index"].notna().any() else np.nan
    q_high = float(grouped["tm_opponent_attendance_index"].quantile(0.67)) if grouped["tm_opponent_attendance_index"].notna().any() else np.nan

    def _tier(value):
        if pd.isna(value) or not np.isfinite(q_low) or not np.isfinite(q_high):
            return "medium"
        if value <= q_low:
            return "low"
        if value >= q_high:
            return "high"
        return "medium"

    grouped["tm_opponent_strength_tier"] = grouped["tm_opponent_attendance_index"].apply(_tier)
    return grouped[
        [
            "away_team_norm",
            "tm_opponent_attendance_index",
            "tm_opponent_goal_diff_avg",
            "tm_opponent_points_avg",
            "tm_opponent_match_count",
            "tm_opponent_strength_tier",
        ]
    ].copy()


def build_transfermarkt_opponent_priors_from_df(transfermarkt_df, min_matches=2):
    if transfermarkt_df is None or len(transfermarkt_df) == 0:
        return pd.DataFrame(
            columns=[
                "away_team_norm",
                "tm_opponent_attendance_index",
                "tm_opponent_goal_diff_avg",
                "tm_opponent_points_avg",
                "tm_opponent_match_count",
                "tm_opponent_strength_tier",
            ]
        )

    df = transfermarkt_df.copy()
    if "match_date" not in df.columns:
        return pd.DataFrame(columns=["away_team_norm", "tm_opponent_attendance_index", "tm_opponent_goal_diff_avg", "tm_opponent_points_avg", "tm_opponent_match_count", "tm_opponent_strength_tier"])

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if "away_team" not in df.columns:
        if "opponent_name_normalized" in df.columns:
            df["away_team"] = df["opponent_name_normalized"]
        elif "opponent_name_raw" in df.columns:
            df["away_team"] = df["opponent_name_raw"]
        else:
            df["away_team"] = "unknown"

    if "attendance" not in df.columns:
        df["attendance"] = np.nan
    if "goals_for" not in df.columns:
        df["goals_for"] = np.nan
    if "goals_against" not in df.columns:
        df["goals_against"] = np.nan

    df["attendance"] = _safe_numeric(df["attendance"])
    df["goals_for"] = _safe_numeric(df["goals_for"])
    df["goals_against"] = _safe_numeric(df["goals_against"])

    if "is_home_match" in df.columns:
        home_mask = df["is_home_match"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])
        df = df[home_mask].copy()

    df = df.dropna(subset=["match_date"]).copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["away_team_norm", "tm_opponent_attendance_index", "tm_opponent_goal_diff_avg", "tm_opponent_points_avg", "tm_opponent_match_count", "tm_opponent_strength_tier"])

    df["away_team_norm"] = _normalize_team_name(df["away_team"])
    base = df.copy()
    base["goal_diff"] = base["goals_for"] - base["goals_against"]
    base["points"] = _points_from_goal_diff(base["goal_diff"].fillna(0.0))
    return _build_priors_from_base(base, min_matches=min_matches)


def build_transfermarkt_opponent_priors(external_csv_path, min_matches=2):
    df, stats = load_transfermarkt_matches(external_csv_path)
    if len(df) == 0:
        empty = pd.DataFrame(
            columns=[
                "away_team_norm",
                "tm_opponent_attendance_index",
                "tm_opponent_goal_diff_avg",
                "tm_opponent_points_avg",
                "tm_opponent_match_count",
                "tm_opponent_strength_tier",
            ]
        )
        stats.update({"rows_used_for_priors": 0})
        return empty, stats

    base = df.copy()
    base["goal_diff"] = base["goals_for"] - base["goals_against"]
    base["points"] = _points_from_goal_diff(base["goal_diff"].fillna(0.0))
    grouped = _build_priors_from_base(base, min_matches=min_matches)

    stats.update({
        "rows_used_for_priors": int(len(df)),
        "opponents_with_priors": int(len(grouped)),
    })
    return grouped, stats


def merge_transfermarkt_opponent_priors(df, priors_df):
    if len(df) == 0:
        return df.copy()

    out = df.copy()
    out["away_team_norm"] = _normalize_team_name(out.get("away_team", pd.Series(index=out.index, dtype=object)))

    if len(priors_df) == 0:
        for col in [
            "tm_opponent_attendance_index",
            "tm_opponent_goal_diff_avg",
            "tm_opponent_points_avg",
            "tm_opponent_match_count",
            "tm_opponent_strength_tier",
        ]:
            if col not in out.columns:
                out[col] = np.nan if col != "tm_opponent_strength_tier" else "medium"
        return out

    merged = out.merge(priors_df, on="away_team_norm", how="left")
    if "tm_opponent_strength_tier" in merged.columns:
        merged["tm_opponent_strength_tier"] = merged["tm_opponent_strength_tier"].fillna("medium")
    return merged
