# Matchday Attendance Prediction — Full Analysis

## 1. Objective

The goal of this project is to develop a model that predicts matchday attendance for OH Leuven in order to support:

* operational planning (staffing, catering)
* marketing decisions (targeted campaigns)
* commercial strategy (hospitality and pricing)

The model aims not only to predict attendance accurately, but also to identify key drivers of fan behavior.

---

## 2. Methodology

### Models Used

We implemented and compared:

* Linear Regression (interpretable baseline)
* Random Forest (non-linear model)
* XGBoost (advanced boosting model)
* Baseline (average attendance)

---

### Validation Strategy

Due to the small dataset (approximately 50–70 matches), we used:

**Leave-One-Out Cross Validation (LOOCV)**

This ensures:

* robust evaluation
* minimal overfitting
* realistic performance estimates

---

### Evaluation Metrics

We evaluated models using:

* MAE (Mean Absolute Error): average prediction error
* RMSE (Root Mean Squared Error): penalizes large errors
* R² (Coefficient of Determination): explanatory power
* MAPE (Mean Absolute Percentage Error): percentage error

---

## 3. Feature Engineering

### Core Focus: Sporting Features

* `points_last_5`: recent performance
* `wins_last_3`: short-term momentum
* `goal_diff_last_5`: dominance

---

### Match Importance (Engineered Feature)

We created a match importance score combining:

* team performance
* season progression

This captures how meaningful a match is for fans.

---

### Supporting Features

* `is_weekend`: timing effect
* `has_promotion`: marketing influence
* `attendance_lag_1`: fan habit
* `opponent_freq`: proxy for opponent strength


# Data Lineage and Feature Construction

## 1. Data Sources

The variables used in the model are derived from multiple internal and external datasets:

| Source      | Description                  |
| ----------- | ---------------------------- |
| df_match    | Match results, teams, scores |
| df_tickets  | Attendance (tickets_scanned) |
| df_context  | Timing, promotions, calendar |
| df_trends   | Google Trends (fan interest) |
| df_articles | Media coverage               |

---

## 2. Feature Construction

### A. Target Variable

**tickets_scanned**

Source:

* df_tickets

Meaning:
Actual number of spectators present at the match.

---

### B. Sporting Variables

**points_last_5**

Source:

* df_match → result_home

Transformation:

```python
points_map = {'W': 3, 'D': 1, 'L': 0}
df['points'] = df['result_home'].map(points_map)
df['points_last_5'] = df['points'].rolling(5).sum().shift(1)
```

Meaning:
Team performance over the last five matches.

---

**wins_last_3**

Source:

* df_match

Transformation:

```python
df['win'] = (df['result_home'] == 'W').astype(int)
df['wins_last_3'] = df['win'].rolling(3).sum().shift(1)
```

Meaning:
Short-term momentum.

---

**goal_diff_last_5**

Source:

* df_match → goals

Transformation:

```python
df['goal_diff'] = df['goals_home_ft'] - df['goals_away_ft']
df['goal_diff_last_5'] = df['goal_diff'].rolling(5).sum().shift(1)
```

Meaning:
Team dominance over recent matches.

---

### C. Match Importance

Built from:

* sporting performance (points_last_5)
* season timing (matchday)

Transformation:

```python
df['season_progress'] = df['matchday'] / df['matchday'].max()
```

Combined into:

```python
df['match_importance'] = ...
```

Meaning:
Represents how meaningful a match is within the season context.

---

### D. Opponent Features

**opponent_freq**

Source:

* df_match → away_team

Transformation:

```python
df['opponent_freq'] = df['away_team'].map(df['away_team'].value_counts())
```

Meaning:
Proxy for opponent familiarity or presence in the dataset.

---

**is_top_opponent**

Source:

* manually defined list

Transformation:

```python
top_teams = ["Club Brugge", "Anderlecht", ...]
df['is_top_opponent'] = df['away_team'].isin(top_teams).astype(int)
```

Meaning:
Proxy for opponent attractiveness.

---

### E. Match Attractiveness

Constructed from:

* match importance
* opponent strength
* top opponent indicator

Transformation:

```python
df['match_attractiveness'] = ...
```

Meaning:
Represents the overall appeal of the match.

---

### F. Context Variables

**is_weekend**

Source:

* df_context

Meaning:
Indicates whether the match is played during the weekend.

---

**has_promotion**

Source:

* df_context

Meaning:
Indicates whether a promotion or marketing action is active.

---

### G. Behavioral Variable

**attendance_lag_1**

Source:

* tickets_scanned

Transformation:

```python
df['attendance_lag_1'] = df['tickets_scanned'].shift(1)
```

Meaning:
Attendance of the previous match, capturing fan momentum.

---

## Key Insight

The variables used in the model are not raw data. They are derived through a series of transformations and feature engineering steps to capture meaningful patterns such as:

* team performance
* match importance
* opponent attractiveness
* fan behavior

---

## 4. Base Model Results

| Model             | MAE  | RMSE | R²   | MAPE  |
| ----------------- | ---- | ---- | ---- | ----- |
| Random Forest     | 1287 | 1671 | 0.30 | 20.6% |
| Linear Regression | 1369 | 1731 | 0.25 | 21.3% |
| XGBoost           | 1449 | 1945 | 0.06 | 23.2% |
| Baseline          | 1683 | 2003 | 0.00 | 26.4% |

---

### Key Insights

* All models outperform the baseline, improving prediction error by approximately 400 spectators
* Random Forest performs best overall
* XGBoost underperforms due to dataset size limitations

---

## 5. Prediction Behavior Analysis

### High Attendance Matches

The models consistently underestimate high-attendance matches.

Examples:

* 9331 predicted as approximately 5600 (Random Forest)
* 10723 predicted as approximately 5690 (Random Forest)

---

### Low Attendance Matches

The models overestimate low-attendance matches.

Examples:

* 4241 predicted as approximately 8300 (Random Forest)
* 4036 predicted as approximately 6150 (Random Forest)

---

### Key Insight

The models show a regression-to-the-mean effect, where predictions are biased toward average attendance levels and fail to capture extreme values.

---

## 6. Root Cause

The primary limitation is the absence of a strong variable capturing match attractiveness.

Missing elements include:

* opponent popularity
* rivalry effects
* perceived importance of the opponent

---

## 7. Improved Model: Match Attractiveness

### Concept

We introduced a new feature:

**Match Attractiveness = Sporting Performance × Opponent Appeal**

---

### Components

* match importance
* opponent strength (normalized)
* top opponent indicator
* interaction term: form × opponent

---

### Business Meaning

Fans attend matches based on the overall attractiveness of the fixture, not on isolated factors.

---

## 8. New Model Results

| Model             | MAE  | RMSE | R²   | MAPE  |
| ----------------- | ---- | ---- | ---- | ----- |
| Linear Regression | 1346 | 1650 | 0.32 | 20.8% |
| Random Forest     | 1331 | 1628 | 0.34 | 21.2% |
| XGBoost           | 1436 | 1765 | 0.22 | 23.1% |

---

## 9. Impact of the New Feature

### Improvement in R²

* Random Forest: 0.30 → 0.34
* Linear Regression: 0.25 → 0.32

The model now explains a larger share of attendance variation.

---

### MAE Trade-off

* Slight increase for Random Forest
* Slight improvement for Linear Regression

---

### Key Insight

The improved model better captures variability in attendance, even if average error remains similar.

---

## 10. Interpretation

### Before Improvement

* Model relied on averages
* Poor handling of extreme matches
* Limited understanding of demand drivers

---

### After Improvement

* Model captures interaction effects
* Better representation of fan decision-making
* Improved structure in predictions

---

### Core Insight

Attendance is driven by the interaction between team performance and opponent attractiveness.

---

## 11. Business Implications

### What the Model Enables

**Operational Planning**

* Predict attendance within approximately ±1300 spectators
* Improve staffing and resource allocation

**Marketing Strategy**

* Identify low-demand matches early
* Trigger targeted campaigns

**Revenue Optimization**

* Recognize high-demand matches
* Optimize pricing and hospitality

---

### Limitations

* Difficulty with extreme cases
* Missing richer opponent-related data
* Limited dataset size

---

## 12. Recommendations

### Improve Opponent Features

Include:

* opponent popularity
* rivalry indicators

---

### Expand Dataset

* more seasons
* more matches

---

### Develop Match Segmentation

| Match Type    | Strategy           |
| ------------- | ------------------ |
| High demand   | maximize revenue   |
| Medium demand | maintain strategy  |
| Low demand    | activate marketing |

---

## 13. Final Conclusion

This project demonstrates that matchday attendance can be predicted with reasonable accuracy using sporting and contextual features. The Random Forest model performs best and significantly outperforms the baseline. The introduction of a match attractiveness feature improves the model’s explanatory power and highlights that attendance is driven by the interaction between team performance and opponent appeal.

While limitations remain due to data constraints, the model provides valuable insights and can support operational and commercial decision-making at OH Leuven.

---

## Final Message

Attendance follows identifiable patterns driven by performance and match attractiveness. By leveraging these insights, OH Leuven can move from reactive to proactive planning.




Sporting:
- points_last_5
- wins_last_3
- goal_diff_last_5
- match_importance
- is_high_importance

Context:
- is_weekend
- season_progress
- opponent_freq
- has_promotion
- attendance_lag_1


The model struggles with:
* Big matches (underestimated)
- High profile games are underpredicted
* Low-demand matches (overestimated)





