# OHL Football Attendance Prediction

This project predicts OHL home match attendance (`tickets_scanned`) with a reproducible machine learning pipeline, CLI workflows, and a Streamlit demo app.

## Core capabilities

- Training pipeline with model comparison and artifact saving
- Prediction pipeline using saved best model metadata
- Reporting pipeline with summary outputs
- Minimal inference contract:
  - `match_date`
  - `away_team`
  - `stage`
  - `kickoff_time`
- Optional Open-Meteo weather enrichment for training and prediction

## Data location

Default data directory:

`C:\Users\ASUS\Desktop\International Project\Data`

Expected files:

- `gold_match.csv`
- `gold_match_tickets.csv`
- `gold_match_context.csv`
- `gold_google_trends_daily.csv`
- `gold_belga_press_articles.csv`
- `gold_match_goals.csv`

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Weather API integration

The project supports Open-Meteo in two modes:

- Historical mode for observed matches in training
- Forecast mode for future matches in prediction

Weather features:

- `weather_temp_mean_c`
- `weather_precipitation_mm`
- `weather_rain_mm`
- `weather_windspeed_max_kmh`
- `weather_bad_flag`

Weather is optional and disabled by default.

### Leakage safety

- Training uses historical daily weather for past observed match dates
- Prediction uses forecast weather for future input rows
- No post-match weather values are used in feature generation

### Caching

Weather responses are cached under:

- `data_cache/weather/`

Cache keys are date and mode specific (`historical` or `forecast`) with stadium location and timezone.

## Train models

Standard training:

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data"
```

Training with weather enrichment:

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data" --use-weather-api
```

Compare weather impact:

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data" --compare-weather-impact
```

Weather impact output:

- `outputs/weather_impact_comparison.csv`

Optional feature minimization experiment:

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data" --run-feature-minimization
```

## Predict from CSV

Input CSV columns remain minimal:

- `match_date`
- `away_team`
- `stage`
- `kickoff_time`

Prediction without weather API:

```powershell
python -m src.predict --input-file "data_examples\new_match_input_template.csv" --data-dir "C:\Users\ASUS\Desktop\International Project\Data"
```

Prediction with weather API:

```powershell
python -m src.predict --input-file "data_examples\new_match_input_template.csv" --data-dir "C:\Users\ASUS\Desktop\International Project\Data" --use-weather-api
```

Output file:

- `outputs/predictions/new_match_predictions.csv`

## Generate report

```powershell
python -m src.report
```

## Run web app

```powershell
streamlit run app.py
```

The app keeps the same minimal input form and uses automatic internal feature generation. Weather information is shown after prediction when weather API mode is enabled by configuration.

## Web smoke test

```powershell
python -m src.web_smoke_test
```

## Main generated artifacts

- `models/best_attendance_model.joblib`
- `models/best_attendance_model_metadata.json`
- `outputs/model_comparison.csv`
- `outputs/weather_impact_comparison.csv` (if weather comparison was run)
- `outputs/minimal_feature_results.csv` (if minimization was run)
- `outputs/feature_importance/best_model_feature_importance.csv`
- `outputs/predictions/test_predictions.csv`
- `outputs/predictions/test_predictions_real_data_detailed.csv`
- `outputs/predictions/top_error_cases.csv`
- `outputs/predictions/actual_vs_predicted_best_model.png`
- `outputs/predictions/residual_distribution_best_model.png`
- `outputs/reports/summary_report.txt`
