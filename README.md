# OHL Match Attendance Prediction

## Objective

This project predicts OHL home match attendance with `tickets_scanned` as target using advanced tabular ML practices.

The training pipeline evaluates:

- Baselines: `mean_baseline`, `opponent_mean_baseline`
- Linear Regression
- Random Forest
- XGBoost (primary candidate)
- CatBoost
- Ensemble (weighted average of XGBoost + CatBoost)

Best model is selected automatically by validation MAE with RMSE as secondary tie-break.

## Real Dataset Location

Default data directory:

`C:\Users\ASUS\Desktop\International Project\Data`

Expected files:

- `gold_match.csv`
- `gold_match_tickets.csv`
- `gold_match_context.csv`
- `gold_google_trends_daily.csv`
- `gold_belga_press_articles.csv`
- `gold_match_goals.csv`

## Advanced Feature Engineering

The feature engineering pipeline uses strict temporal ordering with 70+ leakage-safe features.

**Attendance Lags** (home matches only):
- `attendance_last_match`, `attendance_last_3_avg`, `attendance_last_5_avg`
- `attendance_std_last_3`, `attendance_std_last_5`
- `attendance_ewm` (exponential weighted mean)
- `attendance_last_same_weekday`
- `attendance_vs_same_opponent_last`

**Team Form** (strictly pre-match):
- `points_last_1`, `points_last_3_matches`, `points_last_5_matches`
- `wins_last_1`, `wins_last_3_matches`, `wins_last_5`
- `unbeaten_streak`
- `goals_scored_last_1/3/5`, `goals_conceded_last_1/3/5`
- `goal_difference_last_1/3/5`

**Opponent Strength**:
- `opponent_historical_avg_attendance`
- `opponent_frequency_seen`
- `opponent_avg_goals_scored`, `opponent_avg_goals_conceded`
- `opponent_recent_goals_scored`, `opponent_recent_goals_conceded`
- `big_opponent_flag` (based on historical attendance percentile)
- `opponent_strength_proxy`, `opponent_encoded_rank_proxy`

**Calendar & Schedule**:
- `days_since_previous_home_match`, `days_since_previous_match`
- `consecutive_home_matches`
- `early_season_flag`, `mid_season_flag`, `late_season_flag`

**Media Features** (time-window based):
- `num_articles`, `num_articles_1d`, `num_articles_3d`, `num_articles_7d`
- `avg_days_to_match`
- `articles_trend_slope`

**Google Trends**:
- `ohl_interest`, `ohl_interest_3d_avg`, `ohl_interest_7d_avg`
- `ohl_interest_trend`
- `ohl_interest_last_available`

**Interaction Features**:
- `promotion_weekend_interaction`
- `promotion_big_opponent_interaction`
- `weekend_big_opponent_interaction`
- `ohl_interest_big_opponent_interaction`
- `free_ticket_pressure`

All features use only information available before the match kickoff.

## Walk-Forward Validation

The pipeline uses time-aware walk-forward validation instead of random train-test split:

- 4 expanding window folds on training data
- Strict temporal ordering (no shuffling)
- Average validation metrics across folds
- Final test set for reporting
- Ensures reproducibility and prevents data leakage

## Ensemble Approach

Automatically creates weighted ensemble:
- 60% XGBoost + 40% CatBoost
- Evaluated as candidate model
- Provides robustness through model diversity

## Target Transformation

Optional log1p transformation:
- Apply during training for improved distribution
- Apply expm1 during prediction
- Compare both approaches

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training

Standard training (4 validation folds):

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data"
```

With XGBoost hyperparameter tuning:

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data" --tune-xgb
```

With CatBoost and both models tuned:

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data" --tune-xgb --tune-catboost
```

With all models tuned and log target evaluation:

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data" --tune-rf --tune-xgb --tune-catboost --use-log-target
```

With feature group ablation analysis:

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data" --tune-xgb --run-ablation
```

## Best-Model Selection and Artifacts

After training, the best fitted model is saved to:

- `models/best_attendance_model.joblib`
- `models/best_attendance_model_metadata.json`

Metadata includes:

- best model name (XGBoost, CatBoost, Ensemble, etc.)
- tested model names
- feature list (70+ features)
- target column
- log target flag
- validation fold count
- train/test row counts
- metrics for all models
- training timestamp
- data directory

## Prediction for New Matches

Prediction automatically loads best-model files and reconstructs the same feature schema with historical context.

Template input: see `data_examples/new_match_input_template.csv`

```powershell
python -m src.predict --input-file "path/to/new_matches.csv" --data-dir "C:\Users\ASUS\Desktop\International Project\Data"
```

## Output Files

Training generates comprehensive outputs:

- `outputs/model_comparison.csv` - all model metrics
- `outputs/predictions/test_predictions.csv` - test set results
- `outputs/predictions/top_error_cases.csv` - worst predictions
- `outputs/predictions/residual_distribution.png` - error distribution
- `outputs/predictions/attendance_distribution.png` - target distribution
- `outputs/predictions/actual_vs_predicted_best_model.png` - scatter plot
- `outputs/predictions/attendance_over_time_test.png` - time series
- `outputs/feature_importance/best_model_feature_importance.csv` - top 30 features
- `outputs/feature_importance/top_feature_importance.png` - feature bar chart
- `outputs/ablation_results.csv` - (if --run-ablation)
- `outputs/reports/summary_report.txt` - text summary

## Validation Strategy

Walk-forward validation ensures:
- No future information leakage
- Realistic model evaluation
- Time-series aware
- Reproducible results
- All feature engineering is strictly pre-match

## Reproducibility

- Fixed random state (42) for all models
- Deterministic feature engineering
- TimeSeriesSplit for validation (no shuffling)
- Versioned metadata stored with model


- `data_examples/new_match_input_template.csv`

Run prediction:

```powershell
python -m src.predict --input-file "data_examples\new_match_input_template.csv" --data-dir "C:\Users\ASUS\Desktop\International Project\Data"
```

Output:

- `outputs/predictions/new_match_predictions.csv`

## Reporting

Generate report summary:

```powershell
python -m src.report
```

Regenerate key plots from test predictions:

```powershell
python -m src.report --regenerate-plots
```

## Notebook

`notebooks/MACHINE_learning.ipynb` mirrors the improved pipeline and demonstrates:

- training with new feature engineering
- model comparison including XGBoost
- selected best model metadata
- feature importance
- top error cases
- new-match prediction

## Outputs

Main generated files:

- `outputs/model_comparison.csv`
- `outputs/predictions/test_predictions.csv`
- `outputs/predictions/top_error_cases.csv`
- `outputs/predictions/new_match_predictions.csv`
- `outputs/feature_importance/best_model_feature_importance.csv`
- `outputs/predictions/actual_vs_predicted_best_model.png`
- `outputs/feature_importance/top_feature_importance.png`
- `outputs/predictions/attendance_over_time_test.png`
- `outputs/predictions/attendance_distribution.png`
- `outputs/predictions/residual_distribution_best_model.png`
- `outputs/reports/summary_report.txt`

