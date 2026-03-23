# OHL Attendance Prediction

Predicting match-day attendance for OH Leuven using a Random Forest model — so the club knows when to prep the fireworks.

## Goal

Predict `tickets_scanned` (actual turnstile count) for each home match before it happens.

## Data

Six CSV files provided by OH Leuven, all linked by `match_id`:

- `gold_match.csv` — match info, results, attendance
- `gold_match_tickets.csv` — ticket sales (B2C, B2B, season passes, per tribune)
- `gold_match_context.csv` — weather, promotions, weekday, school holidays
- `gold_match_goals.csv` — goal events per match
- `gold_google_trends_daily.csv` — daily Google interest in OHL
- `gold_belga_press_articles.csv` — press articles with days-to-match proximity

> Data files are not tracked by git. Drop the CSVs into `data/raw/` manually after cloning.

## Repo Structure

```
data/
  raw/          # original CSVs (don't touch)
  processed/    # Cleaned ready for modeling
notebooks/      # EDA → feature engineering → model training
src/            # data_prep.py, model.py, predict.py
outputs/        # Predictions a
```

## Setup

```bash
git clone <repo-url>
cd OHL-Ai-project
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```
