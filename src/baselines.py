import numpy as np
import pandas as pd


def mean_baseline(y_train, size):
    value = float(np.mean(y_train))
    return np.full(shape=size, fill_value=value, dtype=float)


def opponent_mean_baseline(y_train, x_train, x_test, fallback_value):
    train_frame = pd.DataFrame({"away_team": x_train["away_team"], "target": y_train})
    group_means = train_frame.groupby("away_team")["target"].mean()
    predictions = x_test["away_team"].map(group_means).fillna(float(fallback_value)).to_numpy(dtype=float)
    return predictions

