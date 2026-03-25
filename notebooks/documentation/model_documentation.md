# OHL Attendance Prediction — Project Documentation

**UCLL Advanced AI | Project Week March 2026 | Client: OH Leuven**

---

## What we built and why

OHL needs to know how many people are actually going to show up before the match — not after. Right now that planning happens on gut feel. We built a model that predicts `tickets_scanned` (physical gate scans) for every home match so the club can make staffing, catering, and marketing decisions in advance.

We used 71 home matches from the 2022–23 season onwards. That's a small dataset by machine learning standards, so every decision we made was about avoiding overfitting on limited data.

---

## The data

Six CSV files from OHL's gold data layer:

- **gold_match** — match results, kickoff times, the target variable `tickets_scanned`
- **gold_match_tickets** — B2C and B2B ticket sales, season pass holders
- **gold_match_context** — weather, calendar flags, promotions, % free tickets
- **gold_google_trends_daily** — daily Google search interest for "OH Leuven"
- **gold_belga_press_articles** — press articles linked to specific matches
- **gold_match_goals** — goal events per match

One thing worth clarifying: we predict `tickets_scanned`, not `tickets_sold`. The difference matters operationally. Free ticket holders don't always show up. Season pass holders miss games. The scan count is what determines how many stewards, catering portions, and security staff you actually need.

---

## How we built the features

We went through three rounds of feature building.

**Round 1 — the basics.** Sporting form (points and goal difference over the last 5 matches), opponent quality (is it Club Brugge or Genk?), timing (kickoff hour, stage of season), and a 1-match attendance lag. This gave us R² 0.46.

**Round 2 — commercial and buzz signals.** We added pre-match ticket sales (B2C individual buyers, B2B corporate), the percentage of free tickets issued (a direct measure of no-show risk), Google Trends interest in the 7 days before the match, and press article count. We also added rolling 3 and 5-match attendance averages. R² jumped to 0.56.

**Round 3 — cut the noise.** We had 21 opponent one-hot dummy columns (one per team). Combined they contributed 0.4% of the model's predictive power. We dropped all of them. We also ran RFECV — a feature selection algorithm that removes the weakest feature one at a time and checks if the model improves. It settled on 21 features as the sweet spot. R² reached 0.677.

The three features RFECV eliminated that might surprise you: `seasonpass_holders` (barely varies match to match — std of 195 on a mean of 4,321), all the calendar flags like school holidays and weekends (not enough holiday matches in 71 games to learn a reliable pattern), and `match_importance` (absorbed by `match_attractiveness` which already captures the same thing as part of a composite).

---

## How we validated the model

With 71 rows, a standard 80/20 train/test split would be unreliable. We used Leave-One-Out Cross-Validation (LOOCV): hold out one match, train on the other 70, predict the held-out match, repeat 71 times. Every single prediction in our results was made by a model that had never seen that match.

One leakage risk we had to handle carefully: `opponent_avg_attendance_raw` — the historical average attendance when a specific team visits. If we computed that on all 71 matches, the test match's own attendance number would feed into its own prediction. So we recompute it from scratch inside each fold using only the training matches.

---

## Results

| Model | MAE | R² | MAPE |
|---|---|---|---|
| **GradientBoosting** | **870** | **0.677** | **14.2%** |
| Stacking (Ridge + GB) | 896 | 0.670 | 14.3% |
| Ridge | 1,019 | 0.596 | 16.2% |
| XGBoost | 991 | 0.571 | 16.0% |
| RandomForest | 1,069 | 0.554 | 17.4% |

**What these numbers mean in plain language:**

R² of 0.677 means the model explains 67.7% of why attendance varies from match to match. The remaining 32% is things we can't capture — last-minute news, extreme weather, one-off events. On a dataset of 71 matches, this is a solid result.

MAE of 870 means we're off by 870 spectators on average. Average attendance is 6,862, so that's about 13% average error. For a match expected to draw 7,000 people, the model gives you a working range of roughly 6,100 to 7,900.

The client target was R² 0.77, based on an NFL study. That study used 5,055 games. We have 71. We're at 0.677 — 9 points short, and honest about why.

---

## Where the model struggles

65% of predictions land within ±1,000 spectators. But 5 matches (7% of the dataset) were off by more than 2,000. All five have the same explanation:

- **Play-off 2 matches:** Actual attendance was 3,537 and 3,766. The model predicted around 6,200 and 6,500. It has never seen enough dead-atmosphere PO2 fixtures to know how empty they get.
- **Near sell-outs:** Actual was 10,723 and 8,958. The model predicted 7,500 and 6,400. It hasn't seen enough sold-out matches to push predictions that high.

These aren't modelling errors. The dataset just doesn't have enough examples at the extremes. More seasons of data would fix this.

---

## What actually drives attendance

In order of importance from the Random Forest:

1. **Recent attendance history** — the last match, the 3-match average, the 5-match average combined account for about 43% of the model's predictions. Crowd momentum is real and measurable.
2. **Pre-match ticket sales** — B2C individual buyers and B2B corporate bookings together account for roughly 15%. This is useful for OHL: monitoring sales velocity a week out is an early warning system.
3. **Opponent quality** — Club Brugge, Anderlecht, STVV, Mechelen, Westerlo consistently pull bigger crowds. The historical average attendance for each specific opponent adds another layer.
4. **Match context** — stage of season, whether it's a playoff, current form all matter.
5. **Buzz signals** — Google Trends and press article count in the 7 days before the match contribute a small but real amount. Measurable media attention moves the needle.

---

## Overfit check

GradientBoosting fits the training data almost perfectly (Train R² of 0.994). That looks alarming but it's normal for boosting models — they're designed to minimise training error. What matters is the LOOCV number, where every prediction came from a model that hadn't seen that match. 0.677 is the honest figure. Ridge, by contrast, is very clean: train R² of 0.77 vs LOOCV of 0.60.

---

## What would improve it

More data is the honest answer. One more season adds ~25 home matches and would substantially improve generalisation at the extremes. Aside from that:

- **Ticket sales velocity** — not just total B2C sold, but how fast sales are moving day-by-day in the two weeks before the match
- **Season ticket no-show rate per match** — a per-game estimate of how many abonnees will actually scan in
- **Opponent away form** — how the visiting team is playing currently, not just their historical draw
