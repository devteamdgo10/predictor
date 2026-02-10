# trainer/models_registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List

@dataclass
class ModelSpec:
    name: str
    task: str                 # "classification" | "regression"
    family: str               # "linear"|"logistic"|"rf"|"xgb"|"lgbm"|"mlp"|"catboost"
    make_estimator: Callable[[], Any]
    param_distributions: Dict[str, Any]
    supports_class_weight: bool = False
    probabilistic: bool = True
    default_calibrate: bool = False
    # Preparado para futura integración de early stopping / params de fit
    fit_param_grid: Optional[Dict[str, Any]] = None

# ----------------------------
# Hints para espacios range-aware
# ----------------------------
def _size_bucket(hints: Optional[Dict[str, Any]]) -> str:
    """Devuelve 'small' | 'medium' | 'large' según n_samples."""
    if not hints:
        return "small"
    n = int(hints.get("n_samples", 0) or 0)
    if n >= 100_000:
        return "large"
    if n >= 10_000:
        return "medium"
    return "small"

def _rf_space(bucket: str) -> Dict[str, List[Any]]:
    if bucket == "large":
        return {
            "n_estimators": [600, 900, 1200],
            "max_depth": [None, 12, 18, 24, 32],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
    if bucket == "medium":
        return {
            "n_estimators": [500, 800, 1000],
            "max_depth": [None, 10, 16, 24],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
    # small
    return {
        "n_estimators": [300, 500, 800],
        "max_depth": [None, 6, 10, 16, 24],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

def _xgb_space(bucket: str) -> Dict[str, List[Any]]:
    if bucket == "large":
        return {
            "n_estimators": [800, 1200, 1600],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.02, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "gamma": [0.0, 0.1, 0.2],
        }
    if bucket == "medium":
        return {
            "n_estimators": [600, 800, 1200],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.02, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "gamma": [0.0, 0.1, 0.2],
        }
    # small
    return {
        "n_estimators": [500, 800, 1200],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "gamma": [0.0, 0.1, 0.2],
    }

def _lgbm_space(bucket: str) -> Dict[str, List[Any]]:
    if bucket == "large":
        return {
            "n_estimators": [1000, 1400, 1800],
            "num_leaves": [63, 127, 255, 511],
            "max_depth": [-1, 10, 16, 24],
            "learning_rate": [0.02, 0.05, 0.1],
            "feature_fraction": [0.7, 0.8, 1.0],
            "bagging_fraction": [0.7, 0.8, 1.0],
            "bagging_freq": [0, 1],
            "min_child_samples": [10, 20, 40, 80],
            "min_split_gain": [0.0, 0.01, 0.05],
        }
    if bucket == "medium":
        return {
            "n_estimators": [800, 1000, 1400],
            "num_leaves": [31, 63, 127, 255],
            "max_depth": [-1, 6, 10, 16],
            "learning_rate": [0.02, 0.05, 0.1],
            "feature_fraction": [0.7, 0.8, 1.0],
            "bagging_fraction": [0.7, 0.8, 1.0],
            "bagging_freq": [0, 1],
            "min_child_samples": [10, 20, 40, 80],
            "min_split_gain": [0.0, 0.01, 0.05],
        }
    # small
    return {
        "n_estimators": [600, 1000, 1400],
        "num_leaves": [31, 63, 127, 255],
        "max_depth": [-1, 6, 10, 16],
        "learning_rate": [0.02, 0.05, 0.1],
        "feature_fraction": [0.7, 0.8, 1.0],
        "bagging_fraction": [0.7, 0.8, 1.0],
        "bagging_freq": [0, 1],
        "min_child_samples": [10, 20, 40, 80],
        "min_split_gain": [0.0, 0.01, 0.05],
    }

def _catboost_space(bucket: str) -> Dict[str, List[Any]]:
    if bucket == "large":
        return {
            "iterations": [1000, 1500, 2000],
            "depth": [6, 8, 10],
            "learning_rate": [0.02, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 7],
            "bagging_temperature": [0.0, 0.5, 1.0],
            "border_count": [64, 128, 254],
        }
    if bucket == "medium":
        return {
            "iterations": [800, 1200, 1600],
            "depth": [6, 8],
            "learning_rate": [0.02, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5],
            "bagging_temperature": [0.0, 0.5, 1.0],
            "border_count": [64, 128, 254],
        }
    # small
    return {
        "iterations": [600, 1000, 1400],
        "depth": [6, 8],
        "learning_rate": [0.02, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5],
        "bagging_temperature": [0.0, 0.5, 1.0],
        "border_count": [64, 128, 254],
    }

# ----------------------------
# GPU probing (cacheado)
# ----------------------------
_XGB_GPU_AVAILABLE: Optional[bool] = None
_LGBM_GPU_AVAILABLE: Optional[bool] = None
_LGBM_DEVICE_PARAM: Optional[str] = None  # 'device_type' (>=4.0) o 'device' (<=3.x)
_CATBOOST_GPU_AVAILABLE: Optional[bool] = None
_XGB_AVAILABLE: Optional[bool] = None

def _probe_xgb_gpu() -> bool:
    global _XGB_GPU_AVAILABLE
    if _XGB_GPU_AVAILABLE is not None:
        return _XGB_GPU_AVAILABLE
    try:
        from xgboost import XGBClassifier
        X = [[0.0, 0.0], [1.0, 1.0]]
        y = [0, 1]
        clf = XGBClassifier(n_estimators=1, max_depth=1, tree_method="gpu_hist", eval_metric="logloss", verbosity=0)
        clf.fit(X, y)
        _XGB_GPU_AVAILABLE = True
    except Exception:
        _XGB_GPU_AVAILABLE = False
    return _XGB_GPU_AVAILABLE


def _xgb_available() -> bool:
    global _XGB_AVAILABLE
    if _XGB_AVAILABLE is not None:
        return _XGB_AVAILABLE
    try:
        import xgboost  # noqa: F401
        _XGB_AVAILABLE = True
    except Exception:
        _XGB_AVAILABLE = False
    return _XGB_AVAILABLE

def _probe_lgbm_gpu() -> bool:
    global _LGBM_GPU_AVAILABLE, _LGBM_DEVICE_PARAM
    if _LGBM_GPU_AVAILABLE is not None:
        return _LGBM_GPU_AVAILABLE
    try:
        from lightgbm import LGBMClassifier
        tmp = LGBMClassifier()
        params = tmp.get_params()
        dev_param = "device_type" if "device_type" in params else ("device" if "device" in params else None)
        if dev_param is None:
            _LGBM_GPU_AVAILABLE = False
            _LGBM_DEVICE_PARAM = None
            return False
        X = [[0.0, 0.0], [1.0, 1.0]]
        y = [0, 1]
        probe_params = {"n_estimators": 1, "max_depth": 2, dev_param: "gpu", "verbosity": -1}
        clf = LGBMClassifier(**probe_params)
        clf.fit(X, y)
        _LGBM_GPU_AVAILABLE = True
        _LGBM_DEVICE_PARAM = dev_param
    except Exception:
        _LGBM_GPU_AVAILABLE = False
        _LGBM_DEVICE_PARAM = None
    return _LGBM_GPU_AVAILABLE

def _probe_catboost_gpu() -> bool:
    global _CATBOOST_GPU_AVAILABLE
    if _CATBOOST_GPU_AVAILABLE is not None:
        return _CATBOOST_GPU_AVAILABLE
    try:
        from catboost import CatBoostClassifier
        X = [[0.0, 0.0], [1.0, 1.0]]
        y = [0, 1]
        clf = CatBoostClassifier(
            iterations=1, depth=2, learning_rate=0.1,
            task_type="GPU", verbose=False, random_seed=42
        )
        clf.fit(X, y)
        _CATBOOST_GPU_AVAILABLE = True
    except Exception:
        _CATBOOST_GPU_AVAILABLE = False
    return _CATBOOST_GPU_AVAILABLE

# ----------------------------
# XGBoost factories (fallback + silencio)
# ----------------------------
def _xgb_cls(use_gpu: bool):
    from xgboost import XGBClassifier
    use_gpu = use_gpu and _probe_xgb_gpu()
    params = dict(
        n_estimators=800, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", n_jobs=-1,
        tree_method="gpu_hist" if use_gpu else "hist",
        verbosity=0
    )
    return XGBClassifier(**params)

def _xgb_reg(use_gpu: bool):
    from xgboost import XGBRegressor
    use_gpu = use_gpu and _probe_xgb_gpu()
    params = dict(
        n_estimators=800, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
        tree_method="gpu_hist" if use_gpu else "hist",
        verbosity=0
    )
    return XGBRegressor(**params)

# ----------------------------
# LightGBM factories (fallback + silencio)
# ----------------------------
def _lgbm_cls(use_gpu: bool):
    from lightgbm import LGBMClassifier
    params = dict(
        n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1,
        verbosity=-1
    )
    if use_gpu and _probe_lgbm_gpu() and _LGBM_DEVICE_PARAM:
        params[_LGBM_DEVICE_PARAM] = "gpu"
    return LGBMClassifier(**params)

def _lgbm_reg(use_gpu: bool):
    from lightgbm import LGBMRegressor
    params = dict(
        n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1,
        verbosity=-1
    )
    if use_gpu and _probe_lgbm_gpu() and _LGBM_DEVICE_PARAM:
        params[_LGBM_DEVICE_PARAM] = "gpu"
    return LGBMRegressor(**params)

# ----------------------------
# CatBoost factories (opcionales)
# ----------------------------
def _cat_cls(use_gpu: bool):
    from catboost import CatBoostClassifier
    params = dict(
        iterations=1000, learning_rate=0.05, depth=6,
        loss_function="Logloss", random_seed=42, verbose=False
    )
    if use_gpu and _probe_catboost_gpu():
        params["task_type"] = "GPU"
    return CatBoostClassifier(**params)

def _cat_reg(use_gpu: bool):
    from catboost import CatBoostRegressor
    params = dict(
        iterations=1000, learning_rate=0.05, depth=6,
        loss_function="RMSE", random_seed=42, verbose=False
    )
    if use_gpu and _probe_catboost_gpu():
        params["task_type"] = "GPU"
    return CatBoostRegressor(**params)

# ----------------------------
# Registries
# ----------------------------
def get_classification_registry(use_gpu: bool = True,
                                hints: Optional[Dict[str, Any]] = None) -> Dict[str, ModelSpec]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    bucket = _size_bucket(hints)
    reg: Dict[str, ModelSpec] = {}

    reg["logistic"] = ModelSpec(
        name="logistic", task="classification", family="logistic",
        make_estimator=lambda: LogisticRegression(max_iter=2000, solver="lbfgs"),
        param_distributions={"C": [1e-3, 1e-2, 1e-1, 1, 10, 100]},
        supports_class_weight=True, probabilistic=True, default_calibrate=False
    )

    reg["rf"] = ModelSpec(
        name="rf", task="classification", family="rf",
        make_estimator=lambda: RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        param_distributions=_rf_space(bucket),
        supports_class_weight=True, probabilistic=True, default_calibrate=True
    )

    # XGBoost (solo si está instalado)
    if _xgb_available():
        try:
            reg["xgb"] = ModelSpec(
                name="xgb", task="classification", family="xgb",
                make_estimator=lambda: _xgb_cls(use_gpu),
                param_distributions=_xgb_space(bucket),
                supports_class_weight=False, probabilistic=True, default_calibrate=True,
                fit_param_grid={
                    # Para futura integración (no requerido por tu _fit_search actual)
                    # "early_stopping_rounds": [50, 100]  # ejemplo
                }
            )
        except Exception:
            pass

    # LightGBM
    try:
        reg["lgbm"] = ModelSpec(
            name="lgbm", task="classification", family="lgbm",
            make_estimator=lambda: _lgbm_cls(use_gpu),
            param_distributions=_lgbm_space(bucket),
            supports_class_weight=True, probabilistic=True, default_calibrate=True,
            fit_param_grid={
                # Para futura integración:
                # "early_stopping_rounds": [50, 100]
            }
        )
    except Exception:
        pass

    # CatBoost (opcional)
    try:
        from catboost import CatBoostClassifier  # noqa: F401
        reg["catboost"] = ModelSpec(
            name="catboost", task="classification", family="catboost",
            make_estimator=lambda: _cat_cls(use_gpu),
            param_distributions=_catboost_space(bucket),
            supports_class_weight=True, probabilistic=True, default_calibrate=True
        )
    except Exception:
        # CatBoost no instalado -> ignorar sin romper
        pass

    # MLP (SciKeras)
    try:
        from scikeras.wrappers import KerasClassifier  # noqa: F401
        reg["mlp"] = ModelSpec(
            name="mlp", task="classification", family="mlp",
            make_estimator=lambda: None,  # se crea en runtime (train.py)
            param_distributions={
                "hidden_layers": [(128,), (256,), (128, 64), (256, 128)],
                "dropout": [0.0, 0.1, 0.2],
                "batch_size": [64, 128, 256],
                "epochs": [15, 25, 40],
                "learning_rate": [1e-3, 5e-4],
            },
            supports_class_weight=True, probabilistic=True, default_calibrate=False
        )
    except Exception:
        pass

    return reg


def get_regression_registry(use_gpu: bool = True,
                            hints: Optional[Dict[str, Any]] = None) -> Dict[str, ModelSpec]:
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    bucket = _size_bucket(hints)
    reg: Dict[str, ModelSpec] = {}

    reg["linear"] = ModelSpec(
        name="linear", task="regression", family="linear",
        make_estimator=lambda: ElasticNet(max_iter=6000, random_state=42),
        param_distributions={"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], "l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0]},
        supports_class_weight=False, probabilistic=False, default_calibrate=False
    )

    reg["rf"] = ModelSpec(
        name="rf", task="regression", family="rf",
        make_estimator=lambda: RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1),
        param_distributions=_rf_space(bucket),
        supports_class_weight=False, probabilistic=False, default_calibrate=False
    )

    if _xgb_available():
        try:
            reg["xgb"] = ModelSpec(
                name="xgb", task="regression", family="xgb",
                make_estimator=lambda: _xgb_reg(use_gpu),
                param_distributions=_xgb_space(bucket),
                supports_class_weight=False, probabilistic=False, default_calibrate=False,
                fit_param_grid={
                    # "early_stopping_rounds": [50, 100]
                }
            )
        except Exception:
            pass

    try:
        reg["lgbm"] = ModelSpec(
            name="lgbm", task="regression", family="lgbm",
            make_estimator=lambda: _lgbm_reg(use_gpu),
            param_distributions=_lgbm_space(bucket),
            supports_class_weight=False, probabilistic=False, default_calibrate=False,
            fit_param_grid={
                # "early_stopping_rounds": [50, 100]
            }
        )
    except Exception:
        pass

    # CatBoost (opcional)
    try:
        from catboost import CatBoostRegressor  # noqa: F401
        reg["catboost"] = ModelSpec(
            name="catboost", task="regression", family="catboost",
            make_estimator=lambda: _cat_reg(use_gpu),
            param_distributions=_catboost_space(bucket),
            supports_class_weight=False, probabilistic=False, default_calibrate=False
        )
    except Exception:
        pass

    # MLP (SciKeras)
    try:
        from scikeras.wrappers import KerasRegressor  # noqa: F401
        reg["mlp"] = ModelSpec(
            name="mlp", task="regression", family="mlp",
            make_estimator=lambda: None,
            param_distributions={
                "hidden_layers": [(128,), (256,), (128, 64), (256, 128)],
                "dropout": [0.0, 0.1, 0.2],
                "batch_size": [64, 128, 256],
                "epochs": [15, 25, 40],
                "learning_rate": [1e-3, 5e-4],
            },
            supports_class_weight=False, probabilistic=False, default_calibrate=False
        )
    except Exception:
        pass

    return reg
