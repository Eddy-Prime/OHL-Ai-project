# OHL Attendance Prediction — Model Documentation

## Project Overview

Predicting matchday attendance (tickets scanned) for OH Leuven home matches at Den Dreef stadium.  
The model combines ticketing, form, contextual, and opponent features across **71 home matches**.

---

## Data Sources

All data comes from the **gold layer** of the OHL data lakehouse, sourced from 3 operational systems:

| Table                       | System              | Contents                             |
| --------------------------- | ------------------- | ------------------------------------ |
| `gold_match`                | Impect/Opta         | Match results, kickoff times, scores |
| `gold_match_tickets`        | Roboticket + iXpole | Tickets sold (B2C + B2B, deduped)    |
| `gold_match_context`        | Multiple            | Weather, calendar, promotions        |
| `gold_google_trends_daily`  | Google Trends       | OHL search interest per day          |
| `gold_belga_press_articles` | Belga Press         | News articles linked to matches      |

---

## Target Variable

**`tickets_scanned`** — physical gate scans on matchday (from Starnet).  
This is the ground truth for actual attendance, not `tickets_sold_total`, because:

- Scanned tickets include staff, press, and accredited personnel not in ticketing systems
- `tickets_sold_total` only counts commercial sales (Roboticket + iXpole)
- Average difference (sold − scanned): **−74 tickets**, meaning scanned is on average slightly higher

---

## Data Preparation

1. Filter to **home matches only** (`is_home_match = TRUE`) — 71 matches
2. Merge `gold_match` ← `gold_match_tickets` ← `gold_match_context` on `match_id`
3. Aggregate Google Trends by `match_id` (mean interest)
4. Aggregate Belga articles by `match_id` (count)
5. Sort chronologically by `match_date`

---

## Feature Engineering

### Form Features

| Feature            | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `points_last_5`    | Rolling sum of points (W=3, D=1, L=0) over last 5 matches, shifted 1 |
| `wins_last_3`      | Rolling win count over last 3 matches, shifted 1                     |
| `goal_diff_last_5` | Rolling goal difference sum over last 5 matches, shifted 1           |

> NaN values for early-season matches (first 5 rows) are filled with the column mean to retain all 71 rows.

### Match Importance

| Feature              | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `season_progress`    | `matchday / max(matchday)` — how far into the season |
| `form_strength_norm` | Min-max normalized `points_last_5`                   |
| `match_importance`   | `0.5 × form_strength_norm + 0.5 × season_progress`   |
| `is_high_importance` | Binary flag: above median `match_importance`         |

### Opponent Features

| Feature                       | Description                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------- |
| `opponent_freq`               | How many times this opponent appears in the dataset                              |
| `is_top_opponent`             | Binary flag for Club Brugge, Anderlecht, STVV, KV Mechelen, Westerlo             |
| `opponent_strength_norm`      | Min-max normalized `opponent_freq`                                               |
| `opponent_avg_attendance_raw` | Mean tickets sold when this opponent visited (recomputed leak-free per LOO fold) |

### Match Attractiveness (Composite)

```
match_attractiveness = 0.4 × match_importance
                     + 0.4 × opponent_strength_norm
                     + 0.2 × is_top_opponent
```

### Interaction Feature

| Feature           | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `form_x_opponent` | `points_last_5 × is_top_opponent` — form amplified by big opponent |

### Context Features

| Feature            | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `kickoff_hour`     | Hour extracted from `kickoff_time_local`                      |
| `is_weekend`       | Boolean from `gold_match_context`                             |
| `has_promotion`    | Boolean cast to int — any active promo campaign               |
| `academic_week`    | Week in academic calendar                                     |
| `matchday`         | Matchday number                                               |
| `weather_rain_mm`  | Rainfall on matchday (mm)                                     |
| `attendance_lag_1` | Tickets scanned at previous home match (NaN filled with mean) |

---

## Final Feature Set

```python
features_combined = [
    'opponent_avg_attendance_raw',
    'academic_week',
    'matchday',
    'weather_rain_mm',
    'kickoff_hour',
    'points_last_5',
    'wins_last_3',
    'goal_diff_last_5',
    'match_importance',
    'match_attractiveness',
    'form_x_opponent',
    'is_weekend',
    'season_progress',
    'has_promotion',
    'attendance_lag_1',
]
```

---

## Model Training

### Validation Strategy

**Leave-One-Out Cross Validation (LOOCV)** — chosen because the dataset has only 71 rows.  
Each match is held out once as the test set while the remaining 70 are used for training.  
This maximizes training data usage per fold.

### Data Leakage Prevention

`opponent_avg_attendance_raw` is **recomputed inside each LOO fold** using only training matches.  
This prevents the test match's ticket count from leaking into its own feature.

### Preprocessing

A `StandardScaler` is fit on the training fold only and applied to both train and test within each fold.

---

## Results (LOOCV)

| Model               | MAE       | RMSE      | R²       | MAPE      |
| ------------------- | --------- | --------- | -------- | --------- |
| **RandomForest** ✅ | **1,201** | **1,460** | **0.46** | **19.8%** |
| GradientBoosting    | 1,282     | 1,571     | 0.37     | 20.6%     |
| XGBoost             | 1,305     | 1,710     | 0.26     | 21.8%     |
| Ridge               | 1,322     | 1,605     | 0.34     | 21.1%     |
| LinearRegression    | 1,350     | 1,656     | 0.30     | 21.5%     |

**Best model: RandomForest** (`n_estimators=100, max_depth=5`)

The fact that complex models (XGBoost, GradientBoosting) underperform Ridge on 71 rows confirms  
the dataset is too small for deep ensembles — RandomForest's `max_depth=5` provides the right regularization balance.

### Performance Interpretation

| R²        | Interpretation                                                         |
| --------- | ---------------------------------------------------------------------- |
| < 0.30    | Weak — barely better than guessing the mean                            |
| 0.30–0.50 | **Moderate — current level, useful but unreliable for hard decisions** |
| 0.50–0.70 | Good — actionable for operational planning                             |
| > 0.70    | Strong — reliable for business decisions                               |

With MAE ~1,201 on average attendance of ~6,000, predictions are off by roughly **±20% per match**.

---

## Saved Outputs

| File                                | Contents                                                     |
| ----------------------------------- | ------------------------------------------------------------ |
| `model.pkl`                         | Best model, scaler, residual std, feature list, sigmoid flag |
| `opponent_lookup.json`              | Per-opponent average attendance + global fallback            |
| `output/model_results_combined.csv` | Full LOOCV metrics table                                     |

---

## Known Limitations & Next Steps

| Priority  | Action                                                                        | Expected Impact            |
| --------- | ----------------------------------------------------------------------------- | -------------------------- |
| 🔴 High   | Add `ohl_interest`, `article_count`, `pct_free_tickets`, `seasonpass_holders` | R² → 0.55–0.65             |
| 🟡 Medium | Tune RandomForest: `min_samples_leaf=2, n_estimators=300`                     | MAE −50–100                |
| 🟡 Medium | Add more seasons of data                                                      | Most impactful long-term   |
| 🟢 Low    | Sigmoid target normalization (% of capacity)                                  | Marginal for linear models |

> For a university project this is a solid result — working pipeline, proper LOOCV, leak-free features,  
> and justified model selection. For production use, R² ≥ 0.65 would be the target threshold.
