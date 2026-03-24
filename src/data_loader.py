from pathlib import Path
import pandas as pd

from .config import RAW_FILES


def _read_csv(path):
    if path.name == "gold_belga_press_articles.csv":
        return pd.read_csv(path, escapechar="\\", on_bad_lines="skip")
    return pd.read_csv(path)


def get_missing_required_files(data_dir):
    data_dir = Path(data_dir)
    return [filename for filename in RAW_FILES.values() if not (data_dir / filename).exists()]


def validate_required_data_files(data_dir):
    data_dir = Path(data_dir)
    missing = get_missing_required_files(data_dir)
    if len(missing) == 0:
        return

    expected = "\n".join([f"- {name}" for name in RAW_FILES.values()])
    missing_lines = "\n".join([f"- {name}" for name in missing])
    raise FileNotFoundError(
        "Missing required CSV files in repository data folder.\n"
        f"Data folder: {data_dir}\n"
        "Place all required CSV files in this folder.\n"
        f"Missing files:\n{missing_lines}\n"
        f"Required files:\n{expected}"
    )


def load_raw_tables(data_dir):
    data_dir = Path(data_dir)
    validate_required_data_files(data_dir)
    tables = {}
    for key, filename in RAW_FILES.items():
        file_path = data_dir / filename
        tables[key] = _read_csv(file_path)
    return tables

