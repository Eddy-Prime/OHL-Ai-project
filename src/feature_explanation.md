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





