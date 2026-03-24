from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.getenv("OHL_DATA_DIR", r"C:\Users\ASUS\Desktop\International Project\Data"))
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FEATURE_IMPORTANCE_DIR = OUTPUTS_DIR / "feature_importance"
REPORTS_DIR = OUTPUTS_DIR / "reports"
WEATHER_CACHE_DIR = PROJECT_ROOT / "data_cache" / "weather"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_ARTIFACT_PATH = MODELS_DIR / "trained_random_forest.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
NEW_MATCH_PREDICTIONS_PATH = PREDICTIONS_DIR / "new_match_predictions.csv"
DATA_EXAMPLES_DIR = PROJECT_ROOT / "data_examples"
BEST_MODEL_ARTIFACT_PATH = MODELS_DIR / "best_attendance_model.joblib"
BEST_MODEL_METADATA_PATH = MODELS_DIR / "best_attendance_model_metadata.json"
WEATHER_IMPACT_COMPARISON_PATH = OUTPUTS_DIR / "weather_impact_comparison.csv"
EXTERNAL_TRAINING_SUMMARY_PATH = OUTPUTS_DIR / "external_training_summary.csv"
MODEL_SCHEMA_VERSION = "2.0"

TRANSFERMARKT_EXTERNAL_PATH = Path(
    os.getenv(
        "TRANSFERMARKT_EXTERNAL_PATH",
        r"C:\Users\ASUS\PycharmProjects\PythoParser\data_external\intermediate\transfermarkt_matches.csv",
    )
)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT_SECONDS = 10
STADIUM_LATITUDE = float(os.getenv("OHL_STADIUM_LATITUDE", "50.8798"))
STADIUM_LONGITUDE = float(os.getenv("OHL_STADIUM_LONGITUDE", "4.7005"))
STADIUM_TIMEZONE = os.getenv("OHL_STADIUM_TIMEZONE", "Europe/Brussels")
WEATHER_API_ENABLED_DEFAULT = False
BAD_WEATHER_RAIN_THRESHOLD = float(os.getenv("OHL_BAD_WEATHER_RAIN_THRESHOLD", "3.0"))
BAD_WEATHER_WIND_THRESHOLD = float(os.getenv("OHL_BAD_WEATHER_WIND_THRESHOLD", "30.0"))

RAW_FILES = {
    "match": "gold_match.csv",
    "tickets": "gold_match_tickets.csv",
    "context": "gold_match_context.csv",
    "trends": "gold_google_trends_daily.csv",
    "articles": "gold_belga_press_articles.csv",
    "goals": "gold_match_goals.csv",
}

TARGET_COLUMN = "tickets_scanned"
DATE_COLUMN = "match_date"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS_VALIDATION = 4
PRIMARY_MODEL_CANDIDATE = "xgboost"
TRAINED_MODEL_CANDIDATES = ["linear_regression", "random_forest", "xgboost", "catboost"]
SUPPORT_ENSEMBLE = True
ENSEMBLE_WEIGHTS = {"xgboost": 0.6, "catboost": 0.4}

REQUIRED_FEATURE_COLUMNS = [
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
    "has_promotion",
    "promo_tickets_total",
    "pct_free_tickets",
    "seasonpass_holders",
    "num_articles",
    "avg_days_to_match",
    "attendance_last_match",
    "attendance_last_3_avg",
    "attendance_last_5_avg",
    "attendance_rolling_std_3",
    "attendance_ewm",
    "attendance_last_same_weekday",
    "attendance_vs_same_opponent_last",
    "attendance_std_last_5",
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
    "early_season_flag",
    "mid_season_flag",
    "late_season_flag",
    "consecutive_home_matches",
    "num_articles_1d",
    "num_articles_3d",
    "num_articles_7d",
    "articles_trend_slope",
    "ohl_interest",
    "ohl_interest_last_available",
    "ohl_interest_3d_avg",
    "ohl_interest_7d_avg",
    "ohl_interest_trend",
    "promotion_weekend_interaction",
    "promotion_big_opponent_interaction",
    "weekend_big_opponent_interaction",
    "ohl_interest_big_opponent_interaction",
    "free_ticket_pressure",
    "ohl_goals_last_3_matches",
    "opp_goals_last_3_matches",
]

ADDITIONAL_FEATURE_COLUMNS = []

FEATURE_GROUPS = {
    "base": [
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
        "has_promotion",
        "promo_tickets_total",
        "pct_free_tickets",
        "seasonpass_holders",
    ],
    "lag": [
        "attendance_last_match",
        "attendance_last_3_avg",
        "attendance_last_5_avg",
        "attendance_rolling_std_3",
        "attendance_ewm",
        "attendance_last_same_weekday",
        "attendance_vs_same_opponent_last",
        "attendance_std_last_5",
    ],
    "form": [
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
        "goals_conceded_last_3_matches",
        "goals_conceded_last_5",
        "goal_difference_last_1",
        "goal_difference_last_3_matches",
        "goal_difference_last_5",
    ],
    "opponent": [
        "opponent_historical_avg_attendance",
        "opponent_frequency_seen",
        "opponent_avg_goals_scored",
        "opponent_avg_goals_conceded",
        "opponent_strength_proxy",
        "opponent_recent_goals_scored",
        "opponent_recent_goals_conceded",
        "big_opponent_flag",
        "opponent_encoded_rank_proxy",
    ],
    "schedule": [
        "days_since_previous_home_match",
        "days_since_previous_match",
        "early_season_flag",
        "mid_season_flag",
        "late_season_flag",
        "consecutive_home_matches",
    ],
    "media": [
        "num_articles",
        "avg_days_to_match",
        "num_articles_1d",
        "num_articles_3d",
        "num_articles_7d",
        "articles_trend_slope",
    ],
    "trends": [
        "ohl_interest",
        "ohl_interest_last_available",
        "ohl_interest_3d_avg",
        "ohl_interest_7d_avg",
        "ohl_interest_trend",
    ],
    "interactions": [
        "promotion_weekend_interaction",
        "promotion_big_opponent_interaction",
        "weekend_big_opponent_interaction",
        "ohl_interest_big_opponent_interaction",
        "free_ticket_pressure",
        "ohl_goals_last_3_matches",
        "opp_goals_last_3_matches",
    ],
}


