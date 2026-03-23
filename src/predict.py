from pathlib import Path
import argparse
import json
import numpy as np

import joblib
import pandas as pd

from .config import BEST_MODEL_ARTIFACT_PATH, BEST_MODEL_METADATA_PATH, DEFAULT_DATA_DIR, NEW_MATCH_PREDICTIONS_PATH
from .data_loader import load_raw_tables
from .features import build_inference_dataset, prepare_inference_features
from .utils import ensure_directories


def load_model_and_metadata(model_path=BEST_MODEL_ARTIFACT_PATH, metadata_path=BEST_MODEL_METADATA_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    model = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def predict_from_file(
    input_file,
    data_dir=DEFAULT_DATA_DIR,
    model_path=BEST_MODEL_ARTIFACT_PATH,
    metadata_path=BEST_MODEL_METADATA_PATH,
    output_file=NEW_MATCH_PREDICTIONS_PATH,
):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model, metadata = load_model_and_metadata(model_path=model_path, metadata_path=metadata_path)
    raw_input = pd.read_csv(input_path)
    tables = load_raw_tables(data_dir)
    inference_dataset = build_inference_dataset(tables=tables, new_matches_df=raw_input)

    feature_columns = metadata.get("features_used", [])
    if not feature_columns:
        raise ValueError("Model metadata does not contain features_used")

    model_input = prepare_inference_features(inference_dataset, feature_columns)
    predictions = model.predict(model_input)
    if metadata.get("use_log_target", False):
        predictions = np.expm1(np.asarray(predictions, dtype=float))
    predictions = np.maximum(np.asarray(predictions, dtype=float), 0.0)

    result = raw_input.copy()
    result["predicted_attendance"] = predictions

    output_path = Path(output_file)
    ensure_directories([output_path.parent])
    result.to_csv(output_path, index=False)

    summary = {
        "best_model": metadata.get("best_model_name", "unknown"),
        "rows_scored": int(len(result)),
        "prediction_min": float(result["predicted_attendance"].min()),
        "prediction_mean": float(result["predicted_attendance"].mean()),
        "prediction_max": float(result["predicted_attendance"].max()),
        "output_file": str(output_path),
    }
    return result, summary


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
    print(f"Saved: {summary['output_file']}")


if __name__ == "__main__":
    main()

