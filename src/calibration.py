import numpy as np
from sklearn.linear_model import LinearRegression


def _to_valid_arrays(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    return y_true_arr[mask], y_pred_arr[mask]


def fit_multiplicative_calibrator(y_true, y_pred):
    y_true_arr, y_pred_arr = _to_valid_arrays(y_true, y_pred)
    if y_true_arr.size == 0:
        return 1.0
    denom = float(np.sum(y_pred_arr * y_pred_arr))
    if denom <= 0.0:
        return 1.0
    numer = float(np.sum(y_true_arr * y_pred_arr))
    return numer / denom


def apply_multiplicative_calibrator(y_pred, k):
    y_pred_arr = np.asarray(y_pred, dtype=float)
    out = y_pred_arr * float(k)
    return np.maximum(out, 0.0)


def fit_linear_calibrator(y_true, y_pred):
    y_true_arr, y_pred_arr = _to_valid_arrays(y_true, y_pred)
    if y_true_arr.size == 0:
        model = LinearRegression()
        model.coef_ = np.array([1.0], dtype=float)
        model.intercept_ = 0.0
        model.n_features_in_ = 1
        return model
    x = y_pred_arr.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y_true_arr)
    return model


def apply_linear_calibrator(y_pred, model_or_params):
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if hasattr(model_or_params, "predict"):
        out = model_or_params.predict(y_pred_arr.reshape(-1, 1))
    else:
        a = float(model_or_params.get("a", 1.0))
        b = float(model_or_params.get("b", 0.0))
        out = a * y_pred_arr + b
    return np.maximum(np.asarray(out, dtype=float), 0.0)


def linear_calibrator_params(model):
    coef = float(np.asarray(model.coef_, dtype=float).reshape(-1)[0])
    intercept = float(model.intercept_)
    return {"a": coef, "b": intercept}

