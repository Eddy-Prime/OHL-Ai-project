from pathlib import Path
import argparse
import json
import tempfile
import numpy as np

import joblib
import pandas as pd

from .config import BEST_MODEL_ARTIFACT_PATH, BEST_MODEL_METADATA_PATH, DEFAULT_DATA_DIR, NEW_MATCH_PREDICTIONS_PATH
from .data_loader import load_raw_tables
from .features import USER_INPUT_FEATURES, build_inference_dataset, prepare_inference_features
from .utils import ensure_directories


def load_metadata(metadata_path=BEST_MODEL_METADATA_PATH):
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def _predict_single_model(model_path, model_input):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    model = joblib.load(model_path)
    predictions = np.asarray(model.predict(model_input), dtype=float)
    return np.maximum(predictions, 0.0)


def predict_from_file(
    input_file,
    data_dir=DEFAULT_DATA_DIR,
    model_path=BEST_MODEL_ARTIFACT_PATH,
    metadata_path=BEST_MODEL_METADATA_PATH,
    output_file=NEW_MATCH_PREDICTIONS_PATH,
    use_weather_api=False,
):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    metadata = load_metadata(metadata_path=metadata_path)
    raw_input = pd.read_csv(input_path)
    missing_required = [col for col in USER_INPUT_FEATURES if col not in raw_input.columns]
    if missing_required:
        raise ValueError(f"Missing required input columns: {', '.join(missing_required)}")

    tables = load_raw_tables(data_dir)
    inference_dataset, inference_stats = build_inference_dataset(
        tables=tables,
        new_matches_df=raw_input,
        return_stats=True,
        use_weather_api=False,
    )

    feature_columns = metadata.get("features_used", [])
    if not feature_columns:
        raise ValueError("Model metadata does not contain features_used")

    fill_values = metadata.get("feature_fill_values", {})
    model_input = prepare_inference_features(inference_dataset, feature_columns, fill_values=fill_values)

    best_model_name = metadata.get("best_model_name", "xgboost_log")
    raw_predictions = _predict_single_model(model_path=model_path, model_input=model_input)
    predictions = np.maximum(np.asarray(raw_predictions, dtype=float), 0.0)

    result = raw_input.copy()
    result["raw_predicted_attendance"] = np.asarray(raw_predictions, dtype=float)
    result["predicted_attendance"] = predictions

    weather_columns = [
        "weather_temp_mean_c",
        "weather_precipitation_mm",
        "weather_rain_mm",
        "weather_windspeed_max_kmh",
        "weather_bad_flag",
        "weather_source",
    ]
    for col in weather_columns:
        if col in inference_dataset.columns:
            result[col] = inference_dataset[col].values

    weather_fallback_counts = {}
    for col in ["weather_temp_mean_c", "weather_precipitation_mm", "weather_rain_mm", "weather_windspeed_max_kmh", "weather_bad_flag"]:
        if col in feature_columns:
            if col in inference_dataset.columns:
                weather_fallback_counts[col] = int(inference_dataset[col].isna().sum())
            else:
                weather_fallback_counts[col] = int(len(inference_dataset))

    output_path = Path(output_file)
    ensure_directories([output_path.parent])
    result.to_csv(output_path, index=False)

    summary = {
        "best_model": best_model_name,
        "rows_scored": int(len(result)),
        "prediction_min": float(result["predicted_attendance"].min()),
        "prediction_mean": float(result["predicted_attendance"].mean()),
        "prediction_max": float(result["predicted_attendance"].max()),
        "output_file": str(output_path),
        "auto_generated_features": inference_stats.get("auto_generated_features", []),
        "fallback_global_mean": float(inference_stats.get("fallback", {}).get("global_mean", 0.0)),
        "fallback_counts": inference_stats.get("fallback", {}).get("fallback_counts", {}),
        "weather_api_enabled": False,
        "weather_stats": inference_stats.get("weather", {}),
        "weather_fallback_counts": weather_fallback_counts,
    }
    return result, summary


def predict_new_matches(
    new_matches_df,
    data_dir=DEFAULT_DATA_DIR,
    model_path=BEST_MODEL_ARTIFACT_PATH,
    metadata_path=BEST_MODEL_METADATA_PATH,
    output_file=NEW_MATCH_PREDICTIONS_PATH,
    use_weather_api=False,
):
    if isinstance(new_matches_df, pd.DataFrame):
        input_df = new_matches_df.copy()
    else:
        input_df = pd.DataFrame(new_matches_df)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as input_tmp:
        input_path = Path(input_tmp.name)

    try:
        input_df.to_csv(input_path, index=False)
        return predict_from_file(
            input_file=input_path,
            data_dir=data_dir,
            model_path=model_path,
            metadata_path=metadata_path,
            output_file=output_file,
            use_weather_api=False,
        )
    finally:
        if input_path.exists():
            input_path.unlink()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model-path", type=str, default=str(BEST_MODEL_ARTIFACT_PATH))
    parser.add_argument("--metadata-path", type=str, default=str(BEST_MODEL_METADATA_PATH))
    parser.add_argument("--output-file", type=str, default=str(NEW_MATCH_PREDICTIONS_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    _, summary = predict_from_file(
        input_file=args.input_file,
        data_dir=Path(args.data_dir),
        model_path=Path(args.model_path),
        metadata_path=Path(args.metadata_path),
        output_file=Path(args.output_file),
    )
    print(f"Best model: {summary['best_model']}")
    print(f"Rows scored: {summary['rows_scored']}")
    print(f"Predicted attendance range: {summary['prediction_min']:.0f} - {summary['prediction_max']:.0f}")
    print(f"Mean predicted attendance: {summary['prediction_mean']:.0f}")
    print(f"Auto-generated features: {', '.join(summary['auto_generated_features'])}")
    print(f"Lag fallback global mean: {summary['fallback_global_mean']:.2f}")
    print(f"Lag fallback counts: {summary['fallback_counts']}")
    print(f"Weather API enabled: {summary['weather_api_enabled']}")
    print(f"Weather fetch stats: {summary['weather_stats']}")
    print(f"Weather fallback counts: {summary['weather_fallback_counts']}")
    print(f"Saved: {summary['output_file']}")


if __name__ == "__main__":
    main()

