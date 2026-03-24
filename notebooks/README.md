# Notebooks

This folder contains the official Jupyter notebooks for the OH Leuven attendance prediction project.

## Main Notebooks

### final_attendance_model.ipynb
**Primary notebook for academic submission and full reproducibility.**

This notebook runs the complete production pipeline from scratch:
1. Loads raw data from `data/` directory
2. Builds features (reuses existing src.features logic)
3. Trains xgboost_log model
4. Evaluates on holdout test set
5. Reports metrics and visualizations

**Requirements:**
- All raw CSV files must be present in `data/`:
  - `gold_match.csv`
  - `gold_match_tickets.csv`
  - `gold_match_context.csv`
  - `gold_google_trends_daily.csv`
  - `gold_belga_press_articles.csv`
  - `gold_match_goals.csv`

**Run:**
```powershell
jupyter notebook notebooks/final_attendance_model.ipynb
```

### demo_attendance_model.ipynb
**Quick demonstration using cached predictions.**

This notebook loads pre-computed test predictions and metrics from previous training runs. No raw data files required.

Perfect for:
- Quickly viewing model performance
- Demonstrating notebook structure
- Understanding output format

**Run:**
```powershell
jupyter notebook notebooks/demo_attendance_model.ipynb
```

Or via command-line execution:
```powershell
python -m jupyter nbconvert --to notebook --execute "notebooks\demo_attendance_model.ipynb" --output-dir "notebooks\executed" --output "demo_attendance_model.executed.ipynb"
```

## Data and Commands

Put all required CSV files in `data/`, then run:

```powershell
python -m src.train --check-data-only
python -m src.train
python -m src.predict --input-file "data_examples\new_match_input_template.csv"
```

## Executed Notebooks

Executed notebook outputs are stored in the `executed/` subfolder:

- `executed/final_attendance_model.executed.ipynb`
- `executed/demo_attendance_model.executed.ipynb`

These are for reference only and are not needed for submission.

## Quick Start

- **Have data?** → Use `final_attendance_model.ipynb`
- **Demo only?** → Use `demo_attendance_model.ipynb`
- **Prefer CLI?** → Use `python -m src.train` or `python -m src.predict`

