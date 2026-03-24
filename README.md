# OHL Football Attendance Prediction

This project predicts OHL home match attendance (`tickets_scanned`) using a reproducible machine learning pipeline and a new Streamlit web interface.

## What is already included

- Training pipeline with model comparison and artifact saving
- Prediction pipeline using saved best model metadata
- Reporting pipeline with summary outputs
- Reduced inference contract with minimal user input:
  - `match_date`
  - `away_team`
  - `stage`
  - `kickoff_time`

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

## Train models

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data"
```

Optional feature minimization experiment:

```powershell
python -m src.train --data-dir "C:\Users\ASUS\Desktop\International Project\Data" --run-feature-minimization
```

## Predict from CSV

Use an input CSV with only these columns:

- `match_date`
- `away_team`
- `stage`
- `kickoff_time`

Run prediction:

```powershell
python -m src.predict --input-file "data_examples\new_match_input_template.csv" --data-dir "C:\Users\ASUS\Desktop\International Project\Data"
```

Output file:

- `outputs/predictions/new_match_predictions.csv`

## Generate report

```powershell
python -m src.report
```

## Run the web app

```powershell
streamlit run app.py
```

The app includes:

- Input form for minimal match fields
- Prediction with MAE-based and median-error-based ranges
- Historical context cards
- Charts from saved outputs and historical trends
- Model quality metrics and error profile
- Feature transparency and minimal-feature summary when available
- Artifact viewer for key CSV outputs
- Prediction CSV export

## Web smoke test

```powershell
python -m src.web_smoke_test
```

## Main generated artifacts

- `models/best_attendance_model.joblib`
- `models/best_attendance_model_metadata.json`
- `outputs/model_comparison.csv`
- `outputs/minimal_feature_results.csv` (if minimization was run)
- `outputs/feature_importance/best_model_feature_importance.csv`
- `outputs/predictions/test_predictions.csv`
- `outputs/predictions/top_error_cases.csv`
- `outputs/predictions/actual_vs_predicted_best_model.png`
- `outputs/predictions/residual_distribution_best_model.png`
- `outputs/reports/summary_report.txt`
