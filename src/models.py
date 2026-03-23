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

    numeric_non_empty_cols = [col for col in numeric_cols if x_train[col].notna().any()]
    numeric_empty_cols = [col for col in numeric_cols if col not in numeric_non_empty_cols]
    categorical_non_empty_cols = [col for col in categorical_cols if x_train[col].notna().any()]
    categorical_empty_cols = [col for col in categorical_cols if col not in categorical_non_empty_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )
    numeric_empty_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    categorical_empty_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_non_empty_cols),
            ("numeric_empty", numeric_empty_pipeline, numeric_empty_cols),
            ("categorical", categorical_pipeline, categorical_non_empty_cols),
            ("categorical_empty", categorical_empty_pipeline, categorical_empty_cols),
        ]
    )
    return preprocessor


def is_catboost_available():
    try:
        import catboost
        return True
    except ImportError:
        return False


def _fit_with_optional_tuning(pipeline, x_train, y_train, tune, param_grid):
    if not tune:
        pipeline.fit(x_train, y_train)
        return pipeline

    n_rows = len(x_train)
    n_splits = 3 if n_rows >= 24 else 2
    if n_rows < 12:
        pipeline.fit(x_train, y_train)
        return pipeline

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def train_linear_regression(x_train, y_train):
    preprocessor = build_preprocessor(x_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, tune=False):
    preprocessor = build_preprocessor(x_train)
    base_model = RandomForestRegressor(
        n_estimators=600,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
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
        "model__n_estimators": [300, 600],
        "model__max_depth": [8, 10, None],
        "model__min_samples_split": [2, 4],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.8],
    }
    return _fit_with_optional_tuning(pipeline, x_train, y_train, tune=tune, param_grid=param_grid)


def train_xgboost(x_train, y_train, tune=False):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required. Install dependencies with: pip install -r requirements.txt") from exc

    preprocessor = build_preprocessor(x_train)
    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
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
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "model__subsample": [0.6, 0.8, 0.95, 1.0],
        "model__colsample_bytree": [0.5, 0.7, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 8],
        "model__reg_alpha": [0, 0.5, 1, 2],
        "model__reg_lambda": [1, 5, 10],
        "model__gamma": [0, 1, 3],
    }
    return _fit_with_optional_tuning(pipeline, x_train, y_train, tune=tune, param_grid=param_grid)


def train_catboost(x_train, y_train, tune=False):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("catboost is required. Install dependencies with: pip install -r requirements.txt") from exc

    preprocessor = build_preprocessor(x_train)
    base_model = CatBoostRegressor(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", base_model),
        ]
    )

    if not tune:
        pipeline.fit(x_train, y_train)
        return pipeline

    param_grid = {
        "model__iterations": [300, 500],
        "model__depth": [4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1],
    }

    n_rows = len(x_train)
    n_splits = 3 if n_rows >= 24 else 2
    if n_rows < 12:
        pipeline.fit(x_train, y_train)
        return pipeline

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_



def get_model_feature_importance(model):
    if isinstance(model, SimpleEnsemble):
        first_model = list(model.models.values())[0]
        return get_model_feature_importance(first_model)
    
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
        predictions = []
        total_weight = 0.0
        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 1.0)
            pred = model.predict(x)
            predictions.append(weight * np.asarray(pred, dtype=float))
            total_weight += weight
        
        ensemble_pred = np.sum(predictions, axis=0) / total_weight
        return ensemble_pred

    def fit(self, x, y):
        for model in self.models.values():
            model.fit(x, y)
        return self

    @property
    def named_steps(self):
        class DummyPreprocessor:
            def get_feature_names_out(self):
                if hasattr(list(self.parent.models.values())[0], 'named_steps'):
                    return list(self.parent.models.values())[0].named_steps['preprocessor'].get_feature_names_out()
                return np.array([f"feature_{i}" for i in range(100)])
        
        class DummyWrapper:
            def __init__(self, parent):
                self.parent = parent
            
            def __getitem__(self, key):
                if key == 'preprocessor':
                    obj = DummyPreprocessor()
                    obj.parent = self.parent
                    return obj
                elif key == 'model':
                    return self
                return None
            
            def get_feature_names_out(self):
                if hasattr(list(self.parent.models.values())[0], 'named_steps'):
                    return list(self.parent.models.values())[0].named_steps['preprocessor'].get_feature_names_out()
                return np.array([f"feature_{i}" for i in range(100)])
        
        return DummyWrapper(self)
