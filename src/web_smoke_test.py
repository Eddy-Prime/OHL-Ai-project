from datetime import date, time
from pathlib import Path

from .config import BEST_MODEL_ARTIFACT_PATH, BEST_MODEL_METADATA_PATH, DEFAULT_DATA_DIR
from .web_helpers import build_single_input_frame, run_prediction_from_frame


def main():
    frame = build_single_input_frame(
        match_date=date.today(),
        away_team="Club Brugge",
        stage="Regular Season",
        kickoff_time=time(hour=20, minute=45),
    )
    pred_df, summary = run_prediction_from_frame(
        input_df=frame,
        data_dir=Path(DEFAULT_DATA_DIR),
        model_path=Path(BEST_MODEL_ARTIFACT_PATH),
        metadata_path=Path(BEST_MODEL_METADATA_PATH),
    )
    print(pred_df.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()

