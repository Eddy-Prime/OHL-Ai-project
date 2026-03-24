import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from .config import RANDOM_STATE


def build_preprocessor(x_train):
    categorical_cols = x_train.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric_cols = [col for col in x_train.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def _fit_with_optional_tuning(pipeline, x_train, y_train, tune, param_grid, sample_weight=None):
    fit_params = {}
    if sample_weight is not None:
        fit_params["model__sample_weight"] = np.asarray(sample_weight, dtype=float)

    if not tune:
        pipeline.fit(x_train, y_train, **fit_params)
        return pipeline

    n_rows = len(x_train)
    n_splits = 3 if n_rows >= 24 else 2
    if n_rows < 12:
        pipeline.fit(x_train, y_train, **fit_params)
        return pipeline

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train, **fit_params)
    return search.best_estimator_


def train_linear_regression(x_train, y_train, sample_weight=None):
    preprocessor = build_preprocessor(x_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    fit_params = {}
    if sample_weight is not None:
        fit_params["model__sample_weight"] = np.asarray(sample_weight, dtype=float)
    model.fit(x_train, y_train, **fit_params)
    return model


def train_residual_model(x_train, y_train):
    return train_linear_regression(x_train=x_train, y_train=y_train)


def train_random_forest(x_train, y_train, tune=False, sample_weight=None):
    preprocessor = build_preprocessor(x_train)
    base_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", base_model),
        ]
    )

    param_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 5, 8],
    }
    return _fit_with_optional_tuning(pipeline, x_train, y_train, tune=tune, param_grid=param_grid, sample_weight=sample_weight)


def train_xgboost(x_train, y_train, tune=False, sample_weight=None):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required. Install dependencies with: pip install -r requirements.txt") from exc

    preprocessor = build_preprocessor(x_train)
    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", base_model),
        ]
    )

    param_grid = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
    }
    return _fit_with_optional_tuning(pipeline, x_train, y_train, tune=tune, param_grid=param_grid, sample_weight=sample_weight)



def get_model_feature_importance(model):
    if isinstance(model, LogTargetModel):
        return get_model_feature_importance(model.base_model)
    if isinstance(model, ResidualCorrectedModel):
        return get_model_feature_importance(model.base_model)
    if isinstance(model, WeightedBlendModel):
        if "xgboost" in model.models:
            return get_model_feature_importance(model.models["xgboost"])
        return get_model_feature_importance(list(model.models.values())[0])
    if isinstance(model, SimpleEnsemble):
        if "xgboost" in model.models:
            return get_model_feature_importance(model.models["xgboost"])
        return get_model_feature_importance(list(model.models.values())[0])
    
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(regressor, "feature_importances_"):
        importance = np.asarray(regressor.feature_importances_, dtype=float)
    elif hasattr(regressor, "coef_"):
        coef = np.asarray(regressor.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef[0]
        importance = np.abs(coef)
    else:
        importance = np.zeros(shape=len(feature_names), dtype=float)

    order = np.argsort(importance)[::-1]
    sorted_names = feature_names[order]
    sorted_values = importance[order]
    return sorted_names, sorted_values


class SimpleEnsemble:
    def __init__(self, models_dict, weights_dict):
        self.models = models_dict
        self.weights = weights_dict

    def predict(self, x):
        xgb_pred = np.asarray(self.models["xgboost"].predict(x), dtype=float)
        rf_pred = np.asarray(self.models["random_forest"].predict(x), dtype=float)
        xgb_weight = float(self.weights.get("xgboost", 0.7))
        rf_weight = float(self.weights.get("random_forest", 0.3))
        return xgb_weight * xgb_pred + rf_weight * rf_pred

    def fit(self, x, y):
        for model in self.models.values():
            model.fit(x, y)
        return self

    @property
    def named_steps(self):
        if "xgboost" in self.models:
            return self.models["xgboost"].named_steps
        return list(self.models.values())[0].named_steps


class WeightedBlendModel:
    def __init__(self, models_dict, weights_dict, calibration_dict=None):
        self.models = models_dict
        self.weights = weights_dict
        self.calibration = calibration_dict or {}

    def _apply_component_calibration(self, model_name, preds):
        from .calibration import apply_linear_calibrator, apply_multiplicative_calibrator

        config = self.calibration.get(model_name, {})
        if not bool(config.get("enabled", False)):
            return np.asarray(preds, dtype=float)

        calibration_type = config.get("type")
        params = config.get("params") or {}
        if calibration_type == "linear":
            return apply_linear_calibrator(y_pred=preds, model_or_params=params)
        if calibration_type == "multiplicative":
            return apply_multiplicative_calibrator(y_pred=preds, k=float(params.get("k", 1.0)))
        return np.asarray(preds, dtype=float)

    def predict(self, x):
        final_pred = np.zeros(shape=len(x), dtype=float)
        for name, model in self.models.items():
            pred = np.asarray(model.predict(x), dtype=float)
            pred = self._apply_component_calibration(name, pred)
            weight = float(self.weights.get(name, 0.0))
            final_pred += weight * pred
        return final_pred

    @property
    def named_steps(self):
        if "xgboost" in self.models:
            return self.models["xgboost"].named_steps
        return list(self.models.values())[0].named_steps


class ResidualCorrectedModel:
    def __init__(self, base_model, residual_model):
        self.base_model = base_model
        self.residual_model = residual_model

    def predict(self, x):
        base_pred = np.asarray(self.base_model.predict(x), dtype=float)
        residual_pred = np.asarray(self.residual_model.predict(x), dtype=float)
        return np.maximum(base_pred + residual_pred, 0.0)

    @property
    def named_steps(self):
        if hasattr(self.base_model, "named_steps"):
            return self.base_model.named_steps
        raise AttributeError("Base model has no named_steps")


class LogTargetModel:
    def __init__(self, base_model):
        self.base_model = base_model

    def predict(self, x):
        pred_log = np.asarray(self.base_model.predict(x), dtype=float)
        pred = np.expm1(pred_log)
        return np.maximum(pred, 0.0)

    @property
    def named_steps(self):
        if hasattr(self.base_model, "named_steps"):
            return self.base_model.named_steps
        raise AttributeError("Base model has no named_steps")


