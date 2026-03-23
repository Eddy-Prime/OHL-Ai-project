from pathlib import Path
import pandas as pd

from .config import RAW_FILES


def _read_csv(path):
    if path.name == "gold_belga_press_articles.csv":
        return pd.read_csv(path, escapechar="\\", on_bad_lines="skip")
    return pd.read_csv(path)


def load_raw_tables(data_dir):
    data_dir = Path(data_dir)
    tables = {}
    for key, filename in RAW_FILES.items():
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")
        tables[key] = _read_csv(file_path)
    return tables

