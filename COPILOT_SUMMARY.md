# COPILOT Summary

## What Was Changed

- Refactored the training pipeline in `src/train.py` to run end-to-end model comparison across `mean_baseline`, `opponent_mean_baseline`, `xgboost_log`, and `catboost_log`.
- Added automatic best-model selection and export of all required artifacts: model, metadata, comparison table, predictions, error cases, feature importance, plots, and summary outputs.
- Rebuilt feature engineering in `src/features.py` with stronger attendance history, team form, time/schedule, opponent, match-importance, media, trend, and weather-derived features.
- Integrated optional Transfermarkt loading in `src/data_loader.py` and robust Transfermarkt prior-feature utilities in `src/external_data.py`.
- Added CatBoost training support and preprocessing compatibility improvements in `src/models.py`.
- Updated prediction flow in `src/predict.py` to keep metadata alignment, include model name in outputs, and honor `use_weather_api` correctly.
- Updated `README.md` to reflect the upgraded workflow and artifacts.

## Transfermarkt Usage

Yes, Transfermarkt data was used.

- It is used as an external prior source through opponent-level features, not as direct target rows in evaluation.
- Added features include:
  - `tm_opponent_attendance_index`
  - `tm_opponent_goal_diff_avg`
  - `tm_opponent_points_avg`
  - `tm_opponent_match_count`
  - `tm_opponent_strength_tier`
- This keeps internal OH Leuven holdout evaluation statistically clean while still leveraging external signal.

## Which Model Won

- Winner: `catboost_log`

## Final Metrics (Internal Time-Based Test)

- MAE: `600.69`
- RMSE: `799.94`
- R2: `0.3985`
- MAPE: `12.20`
- Median Absolute Error: `482.39`

Comparison baseline reference:

- `mean_baseline` MAE: `1523.14`
- Improvement vs mean baseline (MAE): `922.45`

## What Was Tested

- Ran full training pipeline:
  - `python -m src.train`
- Ran example prediction generation:
  - `python -m src.predict --input-file data_examples/new_match_input_template.csv`
- Regenerated report:
  - `python -m src.report`
- Ran web path smoke test:
  - `python -u -m src.web_smoke_test`
- Validated no IDE errors on modified core files.
- Validated leakage guard and strict time split checks inside `src/train.py`.
- Confirmed portable repository-relative paths in source code.

## Known Limitations

- Dataset size is still small, so metric variance between seasonal windows can remain high.
- Some weather values for future inference rows are missing when weather API usage is disabled.
- Feature minimization currently benchmarks with XGBoost-only for speed and comparability, not all candidate models.
- Transfermarkt integration is prior-feature based; direct mixed-target external training is intentionally avoided to reduce comparability risk.

