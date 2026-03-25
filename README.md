# OH Leuven Attendance Prediction

This project trains and serves a football attendance model for OH Leuven using an advanced research pipeline with rolling time-based validation, improved feature engineering, and robust model selection.

## What is included

- Time-based train/test split with strict chronological ordering (no leakage)
- Rolling backtest framework for honest model validation across multiple folds
- Feature engineering for attendance history, team form, schedule, media, trends, weather, and opponent strength
- Optional Transfermarkt integration through opponent prior features (28 opponents with stable metrics)
- Model comparison across baselines, XGBoost, and CatBoost with tuning
- Automatic best-model selection based on rolling validation metrics

## Required data

Place the required files in `data/`:

- `gold_match.csv`
- `gold_match_tickets.csv`
- `gold_match_context.csv`
- `gold_google_trends_daily.csv`
- `gold_belga_press_articles.csv`
- `gold_match_goals.csv`

Optional:

- `transfermarkt_matches.csv` (for opponent strength priors)

## Quick start

```powershell
python -m src.train_research --check-data-only
python -m src.train_research --tune-catboost --n-rolling-folds 5
python -m src.predict --input-file data_examples/new_match_input_template.csv
python -m src.report
```

Run Streamlit app:

```powershell
streamlit run app.py
```

## Main outputs

- Best model artifact: `models/best_attendance_model.joblib`
- Metadata: `models/best_attendance_model_metadata.json`
- Rolling backtest summary: `outputs/rolling_backtest_summary.json`
- Rolling backtest details: `outputs/rolling_backtest_details.csv`
- Model comparison: `outputs/model_comparison.csv`
- Test predictions: `outputs/predictions/test_predictions.csv`
- Top error cases: `outputs/predictions/top_error_cases.csv`
- New predictions: `outputs/predictions/new_match_predictions.csv`
- Feature importance: `outputs/feature_importance/best_model_feature_importance.csv`
- Plots: `outputs/predictions/*.png`, `outputs/plots/*.png`
- Summary report: `outputs/reports/summary_report.txt`
- Research summary: `COPILOT_R2_RESEARCH_SUMMARY.md`

## Notes

- All paths are repository-relative and portable across machines.
- Evaluation is always on OH Leuven internal holdout data (no external datasets in evaluation).
- Transfermarkt data is used as external opponent context features, not as training targets.
- Rolling validation framework enables honest metrics across chronological folds.

