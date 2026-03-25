# OH Leuven Attendance Prediction

This project trains and serves a football attendance model for OH Leuven with a full reproducible pipeline from raw CSV data to prediction and Streamlit usage.

## What is included

- Time-based train/test split with leakage checks
- Feature engineering for attendance history, team form, schedule, media, trends, weather, and opponent strength
- Optional Transfermarkt integration through opponent prior features
- Model comparison across baselines, XGBoost, and CatBoost
- Automatic best-model selection and artifact export

## Required data

Place the required files in `data/`:

- `gold_match.csv`
- `gold_match_tickets.csv`
- `gold_match_context.csv`
- `gold_google_trends_daily.csv`
- `gold_belga_press_articles.csv`
- `gold_match_goals.csv`

Optional:

- `transfermarkt_matches.csv`

## Quick start

```powershell
python -m src.train --check-data-only
python -m src.train
python -m src.predict --input-file data_examples/new_match_input_template.csv
python -m src.report
```

Optional tuning:

```powershell
python -m src.train --tune-xgb --tune-catboost
```

Run Streamlit app:

```powershell
streamlit run app.py
```

## Main outputs

- Best model artifact: `models/best_attendance_model.joblib`
- Metadata: `models/best_attendance_model_metadata.json`
- Model comparison: `outputs/model_comparison.csv`
- Test predictions: `outputs/predictions/test_predictions.csv`
- Top error cases: `outputs/predictions/top_error_cases.csv`
- New predictions: `outputs/predictions/new_match_predictions.csv`
- Feature importance: `outputs/feature_importance/best_model_feature_importance.csv`
- Plots: `outputs/predictions/*.png`, `outputs/plots/*.png`
- Summary report: `outputs/reports/summary_report.txt`

## Notes

- Paths are repository-relative and portable.
- Evaluation is always on OH Leuven internal holdout data.
- Transfermarkt data is used as an external opponent prior source, not as direct target rows in evaluation.
