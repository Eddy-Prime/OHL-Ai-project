# OH Leuven Attendance Prediction (Final, Self-Contained)

This project is simplified to one final deterministic model: `xgboost_log`.

## Quick Start

Put all required CSV files in the repository `data/` folder, then run:

```powershell
python -m src.train --check-data-only
python -m src.train
python -m src.predict --input-file data_examples/new_match_input_template.csv
```

Notebook entrypoints:

```powershell
jupyter notebook notebooks/final_attendance_model.ipynb
jupyter notebook notebooks/demo_attendance_model.ipynb
```


## Self-contained repository layout

- All required raw data must live in `data/`
- Default data path is repository-relative: `PROJECT_ROOT / "data"`
- No absolute `C:\...` paths are used in code

Required files in `data/`:

- `gold_match.csv`
- `gold_match_tickets.csv`
- `gold_match_context.csv`
- `gold_google_trends_daily.csv`
- `gold_belga_press_articles.csv`
- `gold_match_goals.csv`

## Final model

- Model: `xgboost_log`
- Target transform: `log1p(tickets_scanned)`
- Prediction inverse transform: `expm1(...)`
- Split: time-based holdout on OH Leuven internal data


