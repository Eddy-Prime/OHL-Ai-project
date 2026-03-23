from pathlib import Path


def ensure_directories(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def safe_ratio(count, total):
    if total == 0:
        return 0.0
    return float(count) / float(total)

