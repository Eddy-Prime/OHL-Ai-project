from datetime import date, time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import BEST_MODEL_ARTIFACT_PATH, BEST_MODEL_METADATA_PATH, DEFAULT_DATA_DIR, FEATURE_IMPORTANCE_DIR, OUTPUTS_DIR, PREDICTIONS_DIR, WEATHER_API_ENABLED_DEFAULT
from src.web_helpers import (
    build_interpretation,
    build_prediction_export_frame,
    build_single_input_frame,
    compute_context_statistics,
    format_int_like,
    get_improvement_vs_baseline,
    load_historical_match_level,
    load_project_metadata,
    load_selector_options,
    read_optional_csv,
    run_prediction_from_frame,
)


st.set_page_config(page_title="Football Attendance Predictor", layout="wide")


@st.cache_data(show_spinner=False)
def cached_metadata(metadata_path: str):
    return load_project_metadata(Path(metadata_path))


@st.cache_data(show_spinner=False)
def cached_options(data_dir: str):
    return load_selector_options(Path(data_dir))


@st.cache_data(show_spinner=False)
def cached_history(data_dir: str):
    return load_historical_match_level(Path(data_dir))


@st.cache_data(show_spinner=False)
def cached_csv(path: str):
    return read_optional_csv(Path(path))


metadata = cached_metadata(str(BEST_MODEL_METADATA_PATH))
away_options, stage_options = cached_options(str(DEFAULT_DATA_DIR))
history_df = cached_history(str(DEFAULT_DATA_DIR))

best_model_name = str(metadata.get("best_model_name", "unknown"))
best_metrics = metadata.get("best_metrics", {})
mae = float(best_metrics.get("mae", 0.0))
median_abs_error = float(best_metrics.get("median_abs_error", 0.0))

st.title("Football Attendance Prediction Demo")
st.write("Predict home match attendance with four simple inputs while the pipeline auto-generates internal temporal features.")

col_h1, col_h2, col_h3 = st.columns(3)
col_h1.metric("Best model", best_model_name)
col_h2.metric("Minimal required inputs", ", ".join(metadata.get("user_input_features", ["match_date", "away_team", "stage", "kickoff_time"])))
col_h3.metric("Calibration enabled", "Yes" if bool(metadata.get("calibration_enabled", False)) else "No")

st.subheader("Match Input")
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        match_date = st.date_input("Match date", value=date.today())
        away_team = st.selectbox("Away team", options=away_options, index=0)
    with col2:
        stage = st.selectbox("Stage", options=stage_options, index=0)
        kickoff_time = st.time_input("Kickoff time", value=time(hour=20, minute=45))
    submitted = st.form_submit_button("Predict attendance", use_container_width=True)

if submitted:
    input_df = build_single_input_frame(match_date=match_date, away_team=away_team, stage=stage, kickoff_time=kickoff_time)
    pred_df, pred_summary = run_prediction_from_frame(
        input_df=input_df,
        data_dir=Path(DEFAULT_DATA_DIR),
        model_path=Path(BEST_MODEL_ARTIFACT_PATH),
        metadata_path=Path(BEST_MODEL_METADATA_PATH),
        use_weather_api=WEATHER_API_ENABLED_DEFAULT,
    )

    predicted = float(pred_df.iloc[0]["predicted_attendance"])
    lower_mae = max(0.0, predicted - mae)
    upper_mae = predicted + mae
    lower_med = max(0.0, predicted - median_abs_error)
    upper_med = predicted + median_abs_error

    st.subheader("Prediction Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted attendance", format_int_like(predicted))
    r2.metric("Lower estimate", format_int_like(lower_mae))
    r3.metric("Upper estimate", format_int_like(upper_mae))
    st.caption(f"MAE range: {format_int_like(lower_mae)} to {format_int_like(upper_mae)} | Median absolute error range: {format_int_like(lower_med)} to {format_int_like(upper_med)}")

    weather_cols = ["weather_temp_mean_c", "weather_rain_mm", "weather_windspeed_max_kmh", "weather_bad_flag"]
    if WEATHER_API_ENABLED_DEFAULT and all(col in pred_df.columns for col in weather_cols):
        st.subheader("Weather Summary")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Temperature (C)", f"{float(pred_df.iloc[0]['weather_temp_mean_c']):.1f}" if pd.notna(pred_df.iloc[0]["weather_temp_mean_c"]) else "N/A")
        w2.metric("Rain (mm)", f"{float(pred_df.iloc[0]['weather_rain_mm']):.1f}" if pd.notna(pred_df.iloc[0]["weather_rain_mm"]) else "N/A")
        w3.metric("Wind max (km/h)", f"{float(pred_df.iloc[0]['weather_windspeed_max_kmh']):.1f}" if pd.notna(pred_df.iloc[0]["weather_windspeed_max_kmh"]) else "N/A")
        w4.metric("Bad weather flag", f"{int(float(pred_df.iloc[0]['weather_bad_flag']))}" if pd.notna(pred_df.iloc[0]["weather_bad_flag"]) else "N/A")

    context = compute_context_statistics(history_df=history_df, match_date=match_date, away_team=away_team)

    st.subheader("Context Comparison")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Predicted", format_int_like(predicted))
    c2.metric("Season average", format_int_like(context["season_avg"]))
    c3.metric("Last home match", format_int_like(context["last_home"]))
    c4.metric("Last 3 home avg", format_int_like(context["last3_avg"]))
    c5.metric("Same opponent avg", format_int_like(context["same_opponent_avg"]))

    st.subheader("Charts")
    bar_df = pd.DataFrame(
        {
            "Metric": ["Prediction", "Season Avg", "Last Home", "Last 3 Avg", "Same Opp Avg"],
            "Attendance": [
                predicted,
                context["season_avg"],
                context["last_home"],
                context["last3_avg"],
                context["same_opponent_avg"],
            ],
        }
    )
    st.write("Prediction vs historical averages")
    st.bar_chart(bar_df.set_index("Metric"))

    st.write("Recent attendance trend")
    trend_df = context["recent_trend"][["match_date", "tickets_scanned"]].copy()
    trend_df = trend_df.rename(columns={"tickets_scanned": "attendance"})
    if len(trend_df) > 0:
        st.line_chart(trend_df.set_index("match_date"))
    else:
        st.info("Not enough history to draw a recent trend.")

    image_map = [
        (PREDICTIONS_DIR / "actual_vs_predicted_best_model.png", "Actual vs predicted on test set"),
        (PREDICTIONS_DIR / "residual_distribution_best_model.png", "Residual distribution"),
        (FEATURE_IMPORTANCE_DIR / "top_feature_importance.png", "Feature importance"),
    ]
    image_cols = st.columns(3)
    for idx, (img_path, label) in enumerate(image_map):
        if Path(img_path).exists():
            image_cols[idx].image(str(img_path), caption=label, use_container_width=True)
        else:
            image_cols[idx].info(f"Missing artifact: {Path(img_path).name}")

    st.subheader("Reality-Oriented Statistics")
    improvement = get_improvement_vs_baseline(metadata)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Best model", best_model_name)
    s2.metric("MAE", f"{float(best_metrics.get('mae', float('nan'))):.2f}")
    s3.metric("MAPE", f"{float(best_metrics.get('mape', float('nan'))):.2f}%")
    s4.metric("Median abs error", f"{float(best_metrics.get('median_abs_error', float('nan'))):.2f}")

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("Improvement vs mean baseline (MAE)", f"{improvement:.2f}")
    s6.metric("Feature count", str(len(metadata.get("features_used", []))))
    s7.metric("Calibration enabled", "Yes" if bool(metadata.get("calibration_enabled", False)) else "No")
    s8.metric("Minimal user inputs", str(len(metadata.get("user_input_features", []))))

    st.subheader("Interpretation")
    interpretation = build_interpretation(predicted=predicted, season_avg=context["season_avg"], last3_avg=context["last3_avg"])
    st.write(interpretation)

    st.subheader("Model Transparency")
    top_feature_df = cached_csv(str(FEATURE_IMPORTANCE_DIR / "best_model_feature_importance.csv"))
    if top_feature_df is not None and len(top_feature_df) > 0:
        st.write("Top 5 feature importances")
        st.dataframe(top_feature_df.head(5), use_container_width=True)
    else:
        st.info("Feature importance file is not available.")

    minimal_results_df = cached_csv(str(OUTPUTS_DIR / "minimal_feature_results.csv"))
    if minimal_results_df is not None and len(minimal_results_df) > 0:
        best_min = minimal_results_df.sort_values(["mae", "feature_count"]).head(1)
        st.write("Minimal feature set result")
        st.dataframe(best_min, use_container_width=True)
    else:
        st.info("Minimal feature results file is not available.")

    st.write("Internal lag features are automatically generated from historical home matches before scoring.")

    st.subheader("Error Profile")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Share <= 500", f"{100.0 * float(best_metrics.get('share_abs_error_le_500', 0.0)):.1f}%")
    e2.metric("Share <= 800", f"{100.0 * float(best_metrics.get('share_abs_error_le_800', 0.0)):.1f}%")
    e3.metric("Share <= 1000", f"{100.0 * float(best_metrics.get('share_abs_error_le_1000', 0.0)):.1f}%")
    e4.metric("Max error", format_int_like(best_metrics.get("max_abs_error", float("nan"))))
    e5.metric("Median abs error", format_int_like(best_metrics.get("median_abs_error", float("nan"))))

    st.subheader("Artifact Viewer")
    artifact_paths = {
        "Model comparison": OUTPUTS_DIR / "model_comparison.csv",
        "Minimal feature results": OUTPUTS_DIR / "minimal_feature_results.csv",
        "Feature importance": FEATURE_IMPORTANCE_DIR / "best_model_feature_importance.csv",
        "Test predictions": PREDICTIONS_DIR / "test_predictions.csv",
        "Top error cases": PREDICTIONS_DIR / "top_error_cases.csv",
    }

    for label, path in artifact_paths.items():
        df = cached_csv(str(path))
        with st.expander(label, expanded=False):
            if df is None:
                st.info(f"File not found: {path}")
            else:
                st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Export Prediction")
    export_df = build_prediction_export_frame(
        match_date=match_date,
        away_team=away_team,
        stage=stage,
        kickoff_time=kickoff_time,
        predicted=predicted,
        lower_mae=lower_mae,
        upper_mae=upper_mae,
        lower_med=lower_med,
        upper_med=upper_med,
        best_model=best_model_name,
    )
    st.download_button(
        label="Download prediction as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="attendance_prediction_result.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(f"Auto-generated features during inference: {', '.join(pred_summary.get('auto_generated_features', []))}")
    st.caption(f"Lag fallback usage: {pred_summary.get('fallback_counts', {})}")
else:
    st.info("Enter match details and click Predict attendance.")

