# trainer/train.py
from __future__ import annotations
from .auto_decider import build_auto_plan
from sklearn.base import BaseEstimator, TransformerMixin
from .train_functions import (
    MODEL_STEP,
    _ensure_dir, _timestamped_dir, _save_json, _set_seed,
    _detect_problem_type, _class_imbalance_info, _maybe_add_missing_indicators,
    _auto_select_k, _skewed_numeric_columns,
    _save_corr_heatmap, _save_missing_bar, _try_feature_importance,
    _cv_for_task, _primary_metric_default, _candidate_models, _fit_search,
    _build_mlp_classifier, _build_mlp_regressor,
    export_preprocessed_dataset,
    # para extraer 'pre' incluso si el final es CalibratedClassifierCV
    get_preprocessor_from,
)
from .config import SystemConfig
from .data import prepare_dataframe
from .features import build_preprocessor
from .evaluation import (
    evaluate_classification, evaluate_regression,
    plot_confusion, plot_roc_curve, plot_pr_curve, best_threshold_by_f1
)
from .models_registry import get_classification_registry, get_regression_registry
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, KFold, RandomizedSearchCV, cross_val_predict, cross_val_score
)
from sklearn.calibration import CalibratedClassifierCV

import os
import json
import logging
import warnings
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# --- Fijar backend no interactivo (evita Tkinter y warnings) ---
import matplotlib
import matplotlib.pyplot as plt  # noqa
from sklearn.metrics import f1_score  # NEW: para umbral por f1_macro

matplotlib.use("Agg")

# =========================
# Reducción de ruido global
# =========================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.filterwarnings("ignore", message=".*No further splits with positive gain.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*fitted with feature names.*")
warnings.filterwarnings("ignore", module="lightgbm")
try:
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
except Exception:
    pass
try:
    logging.getLogger("xgboost").setLevel(logging.WARNING)
except Exception:
    pass

# =========================
# Logger único del módulo
# =========================
logger = logging.getLogger("trainer")
if not logger.handlers:
    _h = logger.StreamHandler()
    _fmt = logger.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False  # evita duplicados si el root logger tiene handler

# ====================================
# Flags/switches de comportamiento
# ====================================
DISABLE_MLP: bool = True  # no SciKeras por defecto
ENSEMBLE_TOP_N: int = 1  # mantener 1 para evitar Blend no definido
MAKE_PLOTS: bool = True  # ahora generamos CM/ROC/PR
DO_CALIBRATION: bool = False  # isotónica opcional
ONLY_COMPUTE_METRICS_FOR_FINAL = False  # dejamos como antes

# Fuga/alta cardinalidad sospechosa (lista fija y genérica)
DROP_SUSPECT_LEAKS: bool = True
SUSPECT_LEAK_COLS: List[str] = ["Name", "Ticket", "Cabin"]

# Mejoras automáticas (genéricas y seguras)
AUTO_TWEAKS: bool = True
AUTO_SELECT_K_BEST: bool = True  # setea un k suave si config no lo define
AUTO_POWER_TRANSFORM: bool = True  # activa si hay skew alto
IMPUTE_MISSING_INDICATORS: bool = True  # agrega MI__col para columnas con NaN

# Artefactos extra
SAVE_FEATURE_SCHEMA: bool = True
SAVE_OOF_PREDICTIONS: bool = True  # y_true, y_pred, proba_*
SAVE_CORR_HEATMAP: bool = True
SAVE_MISSING_BAR: bool = True
TOP_FEATURES_CSV: str = "feature_importances.csv"

# Early stopping post-fit (opcional, por defecto OFF → no cambia comportamiento)
EARLYSTOP_POSTFIT: bool = False
EARLY_STOPPING_ROUNDS: int = 50
VALID_SIZE: float = 0.2

# [AUTO-PLAN] Import del planificador automático

try:
    from .ensemble import BlendEnsemble  # opcional si ENSEMBLE_TOP_N > 1
except Exception:
    class BlendEnsemble:
        def __init__(self, estimators, weights=None):
            self.estimators = estimators
            self.weights = weights or [1.0] * len(estimators)

        def fit(self, X, y):
            for est in self.estimators:
                if hasattr(est, "fit"):
                    est.fit(X, y)
            return self

        def predict(self, X):
            preds = []
            for est in self.estimators:
                if hasattr(est, "predict"):
                    preds.append(est.predict(X))
            if not preds:
                raise ValueError("BlendEnsemble sin predict().")
            arr = np.column_stack(preds)
            if np.array_equal(np.unique(arr), np.array([0, 1])):
                return (arr.mean(axis=1) >= 0.5).astype(int)
            return arr.mean(axis=1)

        def predict_proba(self, X):
            probas = []
            for est in self.estimators:
                if hasattr(est, "predict_proba"):
                    probas.append(est.predict_proba(X))
            if not probas:
                raise ValueError("BlendEnsemble sin predict_proba().")
            return np.mean(probas, axis=0)


# ============================================================
# NEW: Wrapper para usar un sklearn.Pipeline como transformador
#      intermedio dentro de imblearn.Pipeline sin romper reglas
# ============================================================
# === Reemplaza la clase _PreWrapper completa por esta ===


class _PreWrapper(BaseEstimator, TransformerMixin):
    """
    Adaptador para usar un sklearn.Pipeline 'pre' como step transformador
    dentro de imblearn.Pipeline. Implementa la interfaz de scikit-learn
    para que clone() funcione (get_params/set_params), y reexpone los
    params del pipeline interno con el prefijo 'pre__'.
    """

    def __init__(self, pre: Pipeline):
        self.pre = pre

    # ---- API de estimator ----
    def get_params(self, deep: bool = True):
        params = {"pre": self.pre}
        if deep and hasattr(self.pre, "get_params"):
            for k, v in self.pre.get_params(deep=True).items():
                params[f"pre__{k}"] = v
        return params

    def set_params(self, **params):
        # Soporta tanto 'pre' directo como 'pre__<subparam>'
        pre_params = {}
        for key, value in params.items():
            if key == "pre":
                self.pre = value
            elif key.startswith("pre__"):
                pre_params[key[len("pre__"):]] = value
            else:
                # Parametro desconocido para el wrapper → ignorar (comportamiento sklearn)
                pass
        if pre_params and hasattr(self.pre, "set_params"):
            self.pre.set_params(**pre_params)
        return self

    # ---- API de transformer ----
    def fit(self, X, y=None):
        self.pre.fit(X, y)
        return self

    def transform(self, X):
        return self.pre.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        # TransformerMixin ya implementa, pero aseguramos coherencia
        return self.pre.fit(X, y, **fit_params).transform(X)

    # (Opcional) reexpone nombres si el pre los tiene
    def get_feature_names_out(self, *args, **kwargs):
        if hasattr(self.pre, "get_feature_names_out"):
            return self.pre.get_feature_names_out(*args, **kwargs)
        raise AttributeError("Wrapped 'pre' does not implement get_feature_names_out")


# =========================
# Helpers locales mínimos
# =========================
def _safe_detect_task(y: pd.Series, override: Optional[str] = None) -> str:
    """
    Detección robusta de tarea:
    - Respeta override.
    - Si _detect_problem_type dice 'regression' pero y es no-numérica con pocas categorías => 'classification'.
    """
    t = _detect_problem_type(y, override=override)
    if t == "regression":
        y_clean = pd.Series(y).dropna()
        if (not pd.api.types.is_numeric_dtype(y_clean)) and (y_clean.nunique() <= 20):
            return "classification"
    return t


def _normalize_target_if_needed(task: str, y: pd.Series) -> Tuple[pd.Series, Optional[Dict[str, int]]]:
    """
    Si task=classification y y no es numérico, mapea etiquetas a 0..K-1 en orden estable.
    Devuelve y_mapeado y el dict de mapping (para persistir).
    """
    if task != "classification":
        return y, None
    y_clean = pd.Series(y).dropna()
    if pd.api.types.is_numeric_dtype(y_clean):
        return y, None
    uniq = list(pd.unique(y_clean))
    uniq_sorted = sorted(uniq, key=lambda v: str(v))
    mapping = {str(lbl): i for i, lbl in enumerate(uniq_sorted)}
    y_mapped = y.map(lambda v: mapping.get(str(v)) if pd.notna(v) else v)
    return y_mapped.astype("Int64").astype("float").astype(int), mapping  # asegura ints


def _save_env_versions(out_path: Path) -> None:
    """Guarda versiones de runtime para trazabilidad."""
    info = {}
    try:
        import sys, sklearn, numpy, pandas
        info.update({
            "python": sys.version,
            "sklearn": getattr(sklearn, "__version__", None),
            "numpy": getattr(numpy, "__version__", None),
            "pandas": getattr(pandas, "__version__", None),
        })
    except Exception:
        pass
    # opcionales
    for modname in ("xgboost", "lightgbm", "scikeras", "tensorflow"):
        try:
            mod = __import__(modname)
            info[modname] = getattr(mod, "__version__", None)
        except Exception:
            info[modname] = None
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _safe_model_dir(base_dir: Path, model_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name.strip()) or "model"
    return _ensure_dir(base_dir / safe_name)


def _save_auto_model_artifacts(
    run_dir: Path,
    model_name: str,
    task: str,
    estimator,
    y_true: pd.Series,
    metrics: Dict[str, Any],
    cv_score: float,
    primary_metric: str,
    y_pred: Optional[np.ndarray],
    y_proba: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    try:
        models_root = _ensure_dir(run_dir / "models")
        model_dir = _safe_model_dir(models_root, model_name)

        model_path = model_dir / "pipeline.joblib"
        joblib.dump(estimator, str(model_path))

        report = {
            "model": model_name,
            "task": task,
            "cv_primary_metric": primary_metric,
            "cv_primary_score": float(cv_score),
            "metrics": metrics,
            "artifacts": {
                "model": str(model_path),
            },
        }
        _save_json(model_dir / "metrics.json", metrics)

        if task == "classification" and y_pred is not None:
            if MAKE_PLOTS:
                try:
                    cm_path = model_dir / "confusion_matrix.png"
                    plot_confusion(y_true, y_pred, str(cm_path))
                    report["artifacts"]["confusion_matrix"] = str(cm_path)
                except Exception:
                    pass
                if y_proba is not None:
                    try:
                        roc_path = model_dir / "roc_curve.png"
                        pr_path = model_dir / "pr_curve.png"
                        plot_roc_curve(y_true, y_proba, str(roc_path))
                        plot_pr_curve(y_true, y_proba, str(pr_path))
                        report["artifacts"]["roc_curve"] = str(roc_path)
                        report["artifacts"]["pr_curve"] = str(pr_path)
                    except Exception:
                        pass

            if SAVE_OOF_PREDICTIONS:
                try:
                    out = model_dir / "oof_predictions.csv"
                    if y_proba is not None:
                        if y_proba.ndim == 1:
                            df_oof = pd.DataFrame({
                                "y_true": y_true,
                                "y_pred": y_pred,
                                "proba_1": y_proba
                            })
                        else:
                            cols = [f"proba_{i}" for i in range(y_proba.shape[1])]
                            df_oof = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
                            for i, c in enumerate(cols):
                                df_oof[c] = y_proba[:, i]
                    else:
                        df_oof = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
                    df_oof.to_csv(out, index=False)
                    report["artifacts"]["oof_predictions"] = str(out)
                except Exception:
                    pass
        elif task == "regression" and y_pred is not None:
            if MAKE_PLOTS:
                try:
                    import matplotlib.pyplot as plt
                    import numpy as _np
                    scatter_path = model_dir / "prediction_scatter.png"
                    residual_path = model_dir / "residuals.png"

                    plt.figure()
                    plt.scatter(y_true, y_pred, alpha=0.6)
                    min_v = float(_np.nanmin([_np.nanmin(y_true), _np.nanmin(y_pred)]))
                    max_v = float(_np.nanmax([_np.nanmax(y_true), _np.nanmax(y_pred)]))
                    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray")
                    plt.xlabel("y_true")
                    plt.ylabel("y_pred")
                    plt.title("Predicted vs True")
                    plt.tight_layout()
                    plt.savefig(scatter_path)
                    plt.close()

                    plt.figure()
                    residuals = _np.asarray(y_true) - _np.asarray(y_pred)
                    plt.hist(residuals, bins=30)
                    plt.xlabel("Residual")
                    plt.ylabel("Count")
                    plt.title("Residuals Distribution")
                    plt.tight_layout()
                    plt.savefig(residual_path)
                    plt.close()

                    report["artifacts"]["prediction_scatter"] = str(scatter_path)
                    report["artifacts"]["residuals"] = str(residual_path)
                except Exception:
                    pass

            if SAVE_OOF_PREDICTIONS:
                try:
                    out = model_dir / "oof_predictions.csv"
                    df_oof = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
                    df_oof.to_csv(out, index=False)
                    report["artifacts"]["oof_predictions"] = str(out)
                except Exception:
                    pass

        _save_json(model_dir / "report.json", report)
        return {"dir": model_dir, "artifacts": report.get("artifacts", {})}
    except Exception:
        return None


# =========================
# Helpers adicionales puntuales (solicitados)
# =========================
def _enrich_report_with_calibration_and_thresholds(final_model, X, y, run_dir: Path):
    """Agrega Brier, curva de calibración y matrices de confusión por umbral al report.json."""
    try:
        from sklearn.metrics import brier_score_loss, confusion_matrix, precision_recall_curve
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt
        import numpy as _np
        import json as _json

        # Probabilidades positivas (binario). Si no hay, salir sin romper.
        proba = final_model.predict_proba(X)
        proba_pos = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()

        brier = brier_score_loss(y, proba_pos)

        frac_pos, mean_pred = calibration_curve(y, proba_pos, n_bins=10, strategy='quantile')
        plt.figure()
        plt.plot([0, 1], [0, 1], '--')
        plt.plot(mean_pred, frac_pos, marker='o')
        plt.title('Calibration Curve')
        plt.xlabel('Mean predicted prob.')
        plt.ylabel('Fraction of positives')
        plt.tight_layout()
        plt.savefig(Path(run_dir) / 'calibration_curve.png')
        plt.close()

        ps, rs, ths = precision_recall_curve(y, proba_pos)
        f1s = 2 * ps * rs / (ps + rs + 1e-12)
        best_idx = int(_np.nanargmax(f1s))
        best_th = float(ths[best_idx - 1]) if best_idx > 0 and (best_idx - 1) < len(ths) else 0.5

        from sklearn.metrics import confusion_matrix as _cm
        cm_05 = _cm(y, (proba_pos >= 0.5)).tolist()
        cm_best = _cm(y, (proba_pos >= best_th)).tolist()

        report_path = Path(run_dir) / 'report.json'
        try:
            data = _json.loads(report_path.read_text(encoding='utf-8'))
        except Exception:
            data = {}
        data.setdefault('extras', {})
        data['extras'].update({
            'brier_score': brier,
            'calibration_curve_png': str(Path(run_dir) / 'calibration_curve.png'),
            'confusion_threshold_0_5': cm_05,
            'best_threshold_by_PR': best_th,
            'confusion_best_threshold': cm_best
        })
        report_path.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception:
        # Silencioso para no romper ejecución si algo falta
        pass


def _postfit_early_stopping_if_enabled(best_estimator, X, y):
    """Early stopping post-fit (opcional) para XGB/LGBM sin tocar la búsqueda. Devuelve nuevo pipeline o None."""
    if not EARLYSTOP_POSTFIT:
        return None
    if not isinstance(best_estimator, Pipeline):
        return None
    if 'pre' not in best_estimator.named_steps or MODEL_STEP not in best_estimator.named_steps:
        return None

    clf = best_estimator.named_steps[MODEL_STEP]
    name = type(clf).__name__.lower()
    if not any(k in name for k in ('xgb', 'lgbm')):
        return None

    from sklearn.base import clone
    from sklearn.model_selection import StratifiedShuffleSplit

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VALID_SIZE, random_state=42)
        (tr_idx, va_idx), = sss.split(X, y)
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        pre = clone(best_estimator.named_steps['pre'])
        pre.fit(X_tr, y_tr)
        Xtr_t = pre.transform(X_tr)
        Xva_t = pre.transform(X_va)

        clf_t = clone(clf)

        if 'xgb' in name:
            clf_t.set_params(n_estimators=2000, eval_metric='logloss')
            clf_t.fit(Xtr_t, y_tr, eval_set=[(Xva_t, y_va)], verbose=False, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            if getattr(clf_t, 'best_iteration', None) is not None:
                clf_t.set_params(n_estimators=clf_t.best_iteration)
        elif 'lgbm' in name:
            clf_t.set_params(n_estimators=2000)
            clf_t.fit(Xtr_t, y_tr, eval_set=[(Xva_t, y_va)], verbose=False, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            if getattr(clf_t, 'best_iteration_', None) is not None:
                clf_t.set_params(n_estimators=clf_t.best_iteration_)
        else:
            return None

        pre_full = clone(pre)
        pipe_full = Pipeline([('pre', pre_full), (MODEL_STEP, clone(clf_t))])
        pipe_full.fit(X, y)
        return pipe_full
    except Exception:
        return None


# =========================
# Entrenamiento principal
# =========================
def train_system(cfg: SystemConfig) -> Dict[str, Any]:
    # Semilla
    _set_seed(getattr(getattr(cfg, "cv", None), "random_state", None))

    # 1) Carga y saneo de datos
    df, y, X = prepare_dataframe(
        cfg.data.csv_path,
        cfg.data.target,
        cfg.data.features,
        cfg.data.drop_id_like,
        cfg.data.id_patterns,
        cfg.data.high_na_threshold,
        cfg.data.sample_rows,
    )

    # Logs básicos del dataset
    try:
        logger.info(f"Dataset shape: X={X.shape}, y={len(y)}")
        dtypes_counts = {str(dt): int((X.dtypes == dt).sum()) for dt in X.dtypes.unique()}
        logger.info(f"Dtypes: {dtypes_counts}")
        null_counts = X.isna().sum().sort_values(ascending=False)
        top_nulls = {k: int(v) for k, v in null_counts.head(10).items() if v > 0}
        if top_nulls:
            logger.info(f"Top columnas con nulos: {top_nulls}")
    except Exception:
        pass

    # 1.1 Indicadores de nulos genéricos
    if IMpute_MISSING_INDICATORS := IMPUTE_MISSING_INDICATORS:
        X, n_mi = _maybe_add_missing_indicators(X)
        if n_mi > 0:
            logger.info(f"Añadidos indicadores de nulos para {n_mi} columna(s).")

    # 1.2 Heurísticas automáticas (genéricas)
    task = _safe_detect_task(y, override=getattr(cfg.train, "task", None))
    logger.info(f"Detected task: {task}")

    # [TARGET MAP] Si clasificación y el target no es numérico, mapear a 0..K-1
    target_mapping: Optional[Dict[str, int]] = None
    if task == "classification":
        y, target_mapping = _normalize_target_if_needed(task, y)

    # drop fugas genéricas
    if DROP_SUSPECT_LEAKS:
        present = [c for c in SUSPECT_LEAK_COLS if c in X.columns]
        if present:
            logger.warning(f"Columnas potencialmente con fuga quitadas: {present}")
            X = X.drop(columns=present, errors="ignore")
            try:
                df.drop(columns=present, errors="ignore", inplace=True)
            except Exception:
                pass

    # Desbalance y skew (para logs + tweaks de config)
    ratio, pos_rate = _class_imbalance_info(y)
    logger.info(f"Imbalance ratio (max/min): {ratio:.2f} | Positive-rate aprox: {pos_rate:.3f}")

    skew_cols = _skewed_numeric_columns(X, skew_thresh=1.0)
    if skew_cols:
        logger.info(f"Columnas numéricas con skew >= 1.0 (ej.): {skew_cols[:10]}{'...' if len(skew_cols) > 10 else ''}")

    # 2) Artefactos / salida
    # todo cambiar por variable
    # run_dir = out_put + '_' + task
    output_dir = getattr(cfg.train, "output_dir", "outputs")
    run_dir = _timestamped_dir(output_dir, prefix=f"{task}")
    logger.info(f"Artifacts output: {run_dir}")

    # Guardar config
    try:
        _save_json(Path(run_dir) / "config.json", asdict(cfg))
    except Exception:
        try:
            _save_json(Path(run_dir) / "config.json", getattr(cfg, "to_dict", lambda: {})())
        except Exception:
            pass

    # Guardar env.json
    try:
        _save_env_versions(Path(run_dir) / "env.json")
    except Exception:
        pass

    # [TARGET MAP] Persistir mapping del target si aplica
    if target_mapping:
        try:
            _save_json(Path(run_dir) / "target_mapping.json", {"mapping": target_mapping})
        except Exception:
            pass

    # 2.1 Log de columnas y esquema
    try:
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)
        logger.info("Columnas utilizadas:\n"
                    f"Total columnas usadas: {len(X.columns)}\n"
                    f"- Numéricas ({len(num_cols)}): {num_cols}\n"
                    f"- Categóricas ({len(cat_cols)}): {cat_cols}\n")
    except Exception:
        num_cols, cat_cols = [], []

    if SAVE_FEATURE_SCHEMA:
        try:
            _save_json(Path(run_dir) / "features.json", {
                "columns": list(X.columns),
                "dtypes": {c: str(X[c].dtype) for c in X.columns}
            })
        except Exception:
            pass

    # 2.2 Gráficas de diagnóstico previas (correlación / nulos)
    try:
        if SAVE_CORR_HEATMAP:
            _save_corr_heatmap(X, Path(run_dir) / "corr_heatmap.png")
        if SAVE_MISSING_BAR:
            _save_missing_bar(X, Path(run_dir) / "missing_bar.png")
    except Exception as e:
        logger.warning(f"No se pudieron generar gráficas de diagnóstico inicial: {e}")

    # [AUTO-PLAN] Generar plan automático y guardarlo
    plan = build_auto_plan(
        X, y,
        task=getattr(cfg.train, "task", None),
        cfg=cfg,
        total_search_iters=int(getattr(cfg.cv, "n_iter_search", getattr(cfg.cv, "n_iter", 30))),
    )
    # Guardado del plan
    _save_json(Path(run_dir) / "auto_plan.json", plan)

    # 3) Registro de modelos
    use_gpu = bool(getattr(cfg.train, "use_gpu_if_available", False))
    reg_cls = get_classification_registry(use_gpu)
    reg_reg = get_regression_registry(use_gpu)
    registry = reg_cls if task == "classification" else reg_reg

    # 4) Selección de candidatos
    mode = getattr(cfg.train, "mode", "auto")
    model_names_cfg = getattr(cfg.train, "model_names", None)
    include_mlp = (not DISABLE_MLP) and bool(getattr(cfg.train, "include_mlp", False))

    if mode == "custom" and model_names_cfg:
        model_names = [m for m in model_names_cfg if m in registry]
        if not model_names:
            raise ValueError("No valid model names provided for custom mode.")
    else:
        model_names = [m for m in _candidate_models(task, include_mlp=include_mlp) if m in registry]

    results: List[Dict[str, Any]] = []
    fitted_models: List[Pipeline] = []

    # 5) CV y métrica (con tweaks automáticos ligeros)
    folds = int(getattr(cfg.cv, "folds", 5))
    shuffle = bool(getattr(cfg.cv, "shuffle", True))
    random_state = int(getattr(cfg.cv, "random_state", 42))
    n_jobs = int(getattr(cfg.cv, "n_jobs", -1))
    n_iter_search = int(getattr(cfg.cv, "n_iter_search", getattr(cfg.cv, "n_iter", 30)))

    cv = _cv_for_task(task, folds, shuffle, random_state)

    primary_metric_cfg = getattr(cfg.train, "primary_metric", None)
    primary_metric = primary_metric_cfg or _primary_metric_default(task)
    n_classes = int(pd.Series(y).dropna().nunique()) if task == "classification" else None
    if task == "classification" and n_classes and n_classes > 2 and primary_metric == "roc_auc":
        primary_metric = "roc_auc_ovr"
        logger.info("Auto-tweak: métrica primaria -> roc_auc_ovr por multiclass.")
    # Si clasif. y desbalance alto y métrica no definida, usamos f1_macro
    if AUTO_TWEAKS and task == "classification" and primary_metric_cfg is None and ratio >= 1.5:
        primary_metric = "f1_macro"
        logger.info("Auto-tweak: métrica primaria -> f1_macro por desbalance.")

    # [AUTO-PLAN] Aplicar sugerencias del plan sin pisar config explícita
    plan_metric = plan.get("primary_metric", None)
    if primary_metric_cfg is None and plan_metric:
        primary_metric = plan_metric

    # Caching del Pipeline (opcional por plan)
    pipeline_memory = None
    try:
        if plan.get("enable_cache", True):
            cache_dir = Path(run_dir) / ".cache"
            pipeline_memory = joblib.Memory(str(cache_dir), verbose=0)
            logger.info("Auto-plan: habilitado Pipeline(memory=.cache)")
    except Exception:
        pipeline_memory = None

    # ====== Pesos globales (para calibración y fallback) ======
    global_sw = None
    if task == "classification":
        try:
            from collections import Counter
            cnt = Counter(y)
            total = len(y)
            w = {k: total / (len(cnt) * v) for k, v in cnt.items()}
            global_sw = pd.Series(y).map(w).values
        except Exception:
            global_sw = None

    # 6) Entrenamiento por modelo
    for mname in model_names:
        spec = registry[mname]
        family = spec.family
        is_mlp = (family == "mlp")

        # Auto tweaks de preprocesamiento si no vienen en config
        select_k_best = getattr(cfg.preprocess, "select_k_best", None)
        if AUTO_TWEAKS and AUTO_SELECT_K_BEST and select_k_best in (None, 0, "auto"):
            # Preferir k del plan si existe
            select_k_best_plan = plan.get("preprocess", {}).get("select_k_best", None)
            if select_k_best_plan is not None:
                select_k_best = select_k_best_plan
                logger.info(f"Auto-plan: select_k_best -> {select_k_best}")
            else:
                k_auto = _auto_select_k(len(X.columns))
                select_k_best = k_auto
                logger.info(f"Auto-tweak: select_k_best -> {k_auto}")

        # ✅ Clamp mínimo para evitar k == n_features (no hace nada si ya es menor)
        if isinstance(select_k_best, int) and select_k_best >= len(X.columns):
            new_k = max(1, len(X.columns) - 1)
            if new_k != select_k_best:
                logger.info(f"Clamp select_k_best: {select_k_best} → {new_k} (d={len(X.columns)})")
            select_k_best = new_k

        scale_for = getattr(cfg.preprocess, "scale_for", None)
        if AUTO_TWEAKS and (not scale_for):
            scale_for = plan.get("preprocess", {}).get("scale_for", ["linear", "logistic", "mlp"])  # no afecta árboles

        power_transform = getattr(cfg.preprocess, "power_transform", False)
        if AUTO_TWEAKS and AUTO_POWER_TRANSFORM and (not power_transform) and len(skew_cols) > 0:
            pt_from_plan = plan.get("preprocess", {}).get("power_transform", True)
            power_transform = bool(pt_from_plan)
            logger.info("Auto-plan: power_transform -> True (skew alto detectado).")

        # Preprocesador
        pre = build_preprocessor(
            X,
            family=family,
            select_k_best=select_k_best,
            scale_for=scale_for,
            one_hot_for=getattr(cfg.preprocess, "one_hot_for", None),
            numeric_imputer=getattr(cfg.preprocess, "numeric_imputer", "median"),
            categorical_imputer=getattr(cfg.preprocess, "categorical_imputer", "most_frequent"),
            drop_low_variance=getattr(cfg.preprocess, "drop_low_variance", False),
            low_variance_threshold=getattr(cfg.preprocess, "low_variance_threshold", 0.0),
            task=task,
            ordinal_mappings=getattr(cfg.preprocess, "ordinal_mappings", None),
            power_transform=power_transform,
            freq_encode_high_card=plan.get("preprocess", {}).get("freq_encode_high_card", True),
            high_card_threshold=plan.get("preprocess", {}).get("high_card_threshold", 50),
        )

        # Estimador
        if is_mlp:
            est = _build_mlp_classifier() if task == "classification" else _build_mlp_regressor()
        else:
            est = spec.make_estimator()

        # Pesos de clase (clasificación) + ajustes por modelo
        fit_params: Dict[str, Any] = {}
        sw_for_fit = None  # NEW: lo decidimos enrutar según resampling
        _want_pos_weight = None  # NEW: pos_weight deseado si NO hay resampling
        _want_is_unbalance = None  # NEW: bandera para LGBM si NO hay resampling

        if task == "classification":
            class_weight_cfg = getattr(cfg.train, "class_weight", None)
            if class_weight_cfg is None:
                cw_plan = plan.get("class_weight", None)
                if cw_plan is not None:
                    class_weight_cfg = cw_plan
                    logger.info(f"Auto-plan: class_weight -> {class_weight_cfg}")
            if AUTO_TWEAKS and ratio >= 1.5 and class_weight_cfg is None:
                class_weight_cfg = "balanced"
                logger.info("Auto-tweak: class_weight -> balanced (desbalance alto).")

            # Intentar usar class_weight nativo
            if class_weight_cfg == "balanced":
                try:
                    est.set_params(class_weight="balanced")
                except Exception:
                    pass

            # Construir vector de sample_weight (aún NO se agrega a fit_params)
            try:
                from collections import Counter
                cnt = Counter(y)
                total = len(y)
                w = {k: total / (len(cnt) * v) for k, v in cnt.items()}
                sw_for_fit = pd.Series(y).map(w).values
            except Exception:
                sw_for_fit = None

            # Pre-calcular ajustes específicos por modelo (sólo si luego NO hay resampling)
            try:
                est_name = type(est).__name__.lower()
                classes, counts = np.unique(pd.Series(y).dropna(), return_counts=True)
                if len(classes) == 2:
                    if 1 in list(classes):
                        pos = int(counts[list(classes).index(1)])
                    else:
                        pos = int(np.min(counts))
                    neg = int(counts.sum() - pos)
                    _want_pos_weight = max(1.0, neg / max(1, pos))
                    _want_is_unbalance = True
            except Exception:
                _want_pos_weight = None
                _want_is_unbalance = None

        # ====== Resampling opcional (SMOTE) sólo si hay desbalance y imblearn disponible ======
        use_resampling = False
        PipeClass = Pipeline
        Resampler = None
        try:
            if task == "classification" and ratio >= 2.0 and plan.get("resample", True):
                from imblearn.pipeline import Pipeline as ImbPipeline
                from imblearn.over_sampling import SMOTE
                PipeClass = ImbPipeline
                Resampler = SMOTE(k_neighbors=5)
                use_resampling = True
        except Exception:
            use_resampling = False
            PipeClass = Pipeline
            Resampler = None

        # ====== Decidir enrutamiento de sample_weight según haya resampling ======
        if task == "classification":
            if use_resampling:
                # a) NO pasar sample_weight al estimador final (longitud cambiará tras SMOTE)
                fit_params.pop(f"{MODEL_STEP}__sample_weight", None)

                # b) (Opcional) Pasar sample_weight al sampler si lo soporta
                if sw_for_fit is not None and Resampler is not None:
                    try:
                        from inspect import signature
                        if 'sample_weight' in signature(Resampler.fit_resample).parameters:
                            fit_params['resample__sample_weight'] = sw_for_fit
                    except Exception:
                        pass

                # c) Neutralizar compensaciones redundantes en el estimador
                try:
                    est_name = type(est).__name__.lower()
                    if "xgb" in est_name and hasattr(est, "set_params"):
                        est.set_params(scale_pos_weight=1.0)  # neutral
                    if "lgbm" in est_name and hasattr(est, "set_params"):
                        p = est.get_params()
                        if "is_unbalance" in p:
                            est.set_params(is_unbalance=False)
                except Exception:
                    pass

            else:
                # Sin resampling: sí pasamos sample_weight al estimador final
                if sw_for_fit is not None:
                    fit_params[f"{MODEL_STEP}__sample_weight"] = sw_for_fit

                # Y aplicamos los ajustes de desbalance en el estimador
                try:
                    est_name = type(est).__name__.lower()
                    if _want_pos_weight is not None and "xgb" in est_name and hasattr(est, "set_params"):
                        est.set_params(scale_pos_weight=float(_want_pos_weight))
                        logger.info(f"[{mname}] XGB scale_pos_weight={float(_want_pos_weight):.3f}")
                    if _want_is_unbalance and "lgbm" in est_name and hasattr(est, "set_params"):
                        p = est.get_params()
                        if "is_unbalance" in p:
                            est.set_params(is_unbalance=True)
                            logger.info(f"[{mname}] LGBM is_unbalance=True")
                except Exception:
                    pass

        # Construcción del Pipeline (con cache opcional)
        steps = []
        if use_resampling and Resampler is not None:
            # ImbPipeline no permite Pipelines intermedios: usamos _PreWrapper
            pre_step = _PreWrapper(pre) if isinstance(pre, Pipeline) else pre
            steps = [("pre", pre_step), ("resample", Resampler), (MODEL_STEP, est)]
        else:
            steps = [("pre", pre), (MODEL_STEP, est)]

        pipe = PipeClass(steps, memory=pipeline_memory)
        if use_resampling:
            logger.info(f"[{mname}] Resampling=SMOTE aplicado (ratio≈{ratio:.2f}) dentro de CV")

        n_jobs_eff = n_jobs if not is_mlp else 1

        # Presupuesto por modelo y estrategia de búsqueda (plan)
        model_iters_plan = plan.get("model_iters", {})
        n_iter_model = int(model_iters_plan.get(mname, n_iter_search))
        search_strategy = plan.get("search_strategy", "random")

        # Early stopping sugerido (registrado, no aplicado en búsqueda por limitación con Pipeline)
        if plan.get("early_stopping", False) and family in ("xgb", "lgbm"):
            logger.info("Auto-plan: early_stopping sugerido (omitido en búsqueda por limitación con Pipeline).")

        # ==== LOG: Setup ====
        try:
            scale_flag = "on" if (
                    scale_for and any(k in (scale_for or []) for k in ["linear", "logistic", "mlp"])) else "off"
            logger.info(
                f"[{mname}] Setup | family={family} | k_best={select_k_best} | power_transform={bool(power_transform)} | "
                f"scale={scale_flag} | freq_encode=True(th={plan.get('preprocess', {}).get('high_card_threshold', 50)})")
            logger.info(
                f"[{mname}] Search | strategy={search_strategy} | n_iter={n_iter_model} | scorer={primary_metric} | folds={folds} | n_jobs={n_jobs_eff}")
        except Exception:
            pass

        # Búsqueda / fit
        import time
        t0 = time.time()
        try:
            fitted, best_est, best_score = _fit_search(
                pipe,
                spec.param_distributions,
                X, y,
                primary_metric,
                cv,
                n_iter_model,
                n_jobs_eff,
                random_state,
                fit_params,
                search_strategy=search_strategy  # type: ignore
            )
        except TypeError:
            # versión sin 'search_strategy'
            fitted, best_est, best_score = _fit_search(
                pipe,
                spec.param_distributions,
                X, y,
                primary_metric,
                cv,
                n_iter_model,
                n_jobs_eff,
                random_state,
                fit_params
            )
        t1 = time.time()
        try:
            est_name = type(best_est.named_steps.get(MODEL_STEP)).__name__
        except Exception:
            est_name = "?"
        logger.info(
            f"[{mname}] Done | best_cv_{primary_metric}={best_score:.6f} | time={t1 - t0:.1f}s | estimator={est_name}")

        # LOG: mejores params del paso model (si existen)
        try:
            best_model = best_est.named_steps.get(MODEL_STEP)
            if hasattr(best_model, "get_params"):
                logger.info(f"[{mname}] Best params (model): {best_model.get_params()}")
        except Exception:
            pass

        # LOG: info SelectKBest ya ajustado
        try:
            pre_fitted = best_est.named_steps.get("pre")
            sel = getattr(pre_fitted, "pre", None)
            if sel is not None and isinstance(sel, Pipeline):
                sel = sel.named_steps.get("selectk", None)
            else:
                sel = getattr(pre_fitted, "named_steps", {}).get("selectk", None) if hasattr(pre_fitted,
                                                                                             "named_steps") else None

            if sel is not None:
                k_req = getattr(sel, "k", None)
                before = len(getattr(sel, "scores_", [])) if hasattr(sel, "scores_") else None
                # columnas finales tras preprocesar (una fila basta para ancho)
                try:
                    Xtr_sample = pre_fitted.transform(X.iloc[:1]) if hasattr(pre_fitted, "transform") else X.iloc[:1]
                    n_final = int(Xtr_sample.shape[1])
                except Exception:
                    n_final = None
                logger.info(
                    f"[{mname}] SelectKBest | k={k_req} | before={before} | selected={n_final} (nombres no disponibles)")
        except Exception:
            pass

        # Métricas por modelo
        metrics: Dict[str, Any] = {}
        y_pred_model = None
        y_proba_model = None
        if not ONLY_COMPUTE_METRICS_FOR_FINAL:
            if task == "classification":
                # y_pred con weights si la versión lo soporta
                try:
                    y_pred_model = cross_val_predict(best_est, X, y, cv=cv, n_jobs=n_jobs_eff,
                                                     method="predict", fit_params=fit_params)
                except TypeError:
                    y_pred_model = cross_val_predict(best_est, X, y, cv=cv, n_jobs=n_jobs_eff, method="predict")
                y_proba_model = None
                try:
                    y_proba_model = cross_val_predict(best_est, X, y, cv=cv, n_jobs=n_jobs_eff,
                                                      method="predict_proba", fit_params=fit_params)
                except TypeError:
                    try:
                        y_proba_model = cross_val_predict(best_est, X, y, cv=cv, n_jobs=n_jobs_eff,
                                                          method="predict_proba")
                    except Exception:
                        y_proba_model = None
                metrics = evaluate_classification(y, y_pred_model, y_proba_model)
                if y_proba_model is not None and (y_proba_model.ndim == 1 or (y_proba_model.ndim == 2 and y_proba_model.shape[1] == 2)):
                    thr, f1b = best_threshold_by_f1(y, y_proba_model)
                    metrics["best_threshold_f1"] = thr
                    metrics["best_threshold_f1_score"] = f1b
            else:
                try:
                    y_pred_model = cross_val_predict(best_est, X, y, cv=cv, n_jobs=n_jobs_eff,
                                                     method="predict", fit_params=fit_params)
                except TypeError:
                    y_pred_model = cross_val_predict(best_est, X, y, cv=cv, n_jobs=n_jobs_eff, method="predict")
                metrics = evaluate_regression(y, y_pred_model)

        model_artifacts_info: Optional[Dict[str, Any]] = None
        if mode in ("auto", "custom"):
            model_artifacts_info = _save_auto_model_artifacts(
                Path(run_dir),
                mname,
                task,
                best_est,
                y,
                metrics,
                float(best_score),
                primary_metric,
                y_pred_model,
                y_proba_model,
            )

        artifacts_dir = None
        if model_artifacts_info:
            artifacts_dir = str(model_artifacts_info.get("dir"))
        graficas = {}
        if model_artifacts_info:
            artifacts = model_artifacts_info.get("artifacts", {})
            graficas.update({
                "matriz_confusion": artifacts.get("confusion_matrix", ""),
                "curva_roc": artifacts.get("roc_curve", ""),
                "pr_curve": artifacts.get("pr_curve", ""),
                "prediction_scatter": artifacts.get("prediction_scatter", ""),
                "residuals": artifacts.get("residuals", ""),
            })
        corr_path = Path(run_dir) / "corr_heatmap.png"
        missing_path = Path(run_dir) / "missing_bar.png"
        calib_path = Path(run_dir) / "calibration_curve.png"
        graficas.update({
            "corr_heatmap": str(corr_path) if corr_path.exists() else "",
            "missing_bar": str(missing_path) if missing_path.exists() else "",
            "curva_calibracion": str(calib_path) if calib_path.exists() else "",
        })

        results.append({
            "model": mname,
            "family": family,
            "cv_primary_score": float(best_score),
            "metrics": metrics,
            "artifacts_dir": artifacts_dir,
            "graficas_dir": artifacts_dir,
            "graficas": graficas,
        })
        fitted_models.append(best_est)

        partial_payload = {
            "task": task,
            "best_models": [
                {
                    "model": r["model"],
                    "family": r["family"],
                    "cv_primary_score": r["cv_primary_score"],
                    "metrics": r["metrics"],
                    "graficas": r.get("graficas", {}),
                    "graficas_dir": r.get("graficas_dir"),
                    "run_dir": run_dir,
                    "labels_info": {"target_mapping": target_mapping} if target_mapping else None,
                }
                for r in results
            ],
        }
        _save_json(Path(run_dir) / "results_partial.json", partial_payload)

    if not results:
        raise RuntimeError(
            "No models were successfully trained. Revisa dependencias (xgboost, lightgbm, scikeras/tensorflow).")

    # 7) Selección top-N y (opcional) ensamble
    results_sorted = sorted(results, key=lambda r: r["cv_primary_score"], reverse=True)
    top_n = max(1, min(ENSEMBLE_TOP_N, len(results_sorted)))
    top_models = results_sorted[:top_n]

    name2model = {r["model"]: m for r, m in zip(results, fitted_models)}
    ensemble_estimators = [name2model[r["model"]] for r in top_models]

    if len(ensemble_estimators) == 1:
        final_est = ensemble_estimators[0]
    else:
        weights = [1.0] * len(ensemble_estimators)
        final_est = BlendEnsemble(ensemble_estimators, weights)

    # ✅ Early stopping post-fit opcional (no altera si flag OFF)
    maybe_final = _postfit_early_stopping_if_enabled(final_est, X, y)
    if maybe_final is not None:
        final_est = maybe_final

    # 8) Calibración (solo si clasificador y sin ensamble)
    do_calib_plan = bool(plan.get("do_calibration", False))
    if task == "classification" and (do_calib_plan or DO_CALIBRATION) and len(ensemble_estimators) == 1:
        try:
            # Calibramos el PIPELINE completo (pre + model), no sólo el estimador.
            if hasattr(final_est, "fit") and (
                    hasattr(final_est, "predict_proba") or hasattr(final_est, "decision_function")):
                # intentar pasar sample_weight global
                try:
                    if global_sw is not None:
                        final_est = CalibratedClassifierCV(final_est, method="isotonic", cv=cv).fit(X, y,
                                                                                                    sample_weight=global_sw)
                    else:
                        final_est = CalibratedClassifierCV(final_est, method="isotonic", cv=cv).fit(X, y)
                except TypeError:
                    final_est = CalibratedClassifierCV(final_est, method="isotonic", cv=cv).fit(X, y)
                logger.info("Calibrated final estimator with isotonic regression.")
        except Exception as e:
            logger.warning(f"Calibration skipped due to error: {e}")

    # 9) Fit final si hiciera falta
    if hasattr(final_est, "fit") and not hasattr(final_est, "classes_"):
        try:
            final_est.fit(X, y)
        except Exception:
            pass

    # 10) Persistencia
    pipe_name = getattr(cfg.train, "save_pipeline_as", "pipeline.joblib")
    report_name = getattr(cfg.train, "save_report_as", "report.json")
    joblib.dump(final_est, str(Path(run_dir) / pipe_name))

    # 10.1) Exportar dataset preprocesado (X transformado + y)
    try:
        # Extraemos 'pre' incluso si el final es CalibratedClassifierCV
        pre_fitted = get_preprocessor_from(final_est)
        if pre_fitted is not None:
            ok = export_preprocessed_dataset(pre_fitted, X, y, Path(run_dir) / "train_preprocessed.csv")
            if ok:
                logger.info(f"Saved preprocessed training dataset: {Path(run_dir) / 'train_preprocessed.csv'}")

                # Logs claros sobre columnas finales y SelectKBest
                try:
                    # Columnas finales del preprocesamiento (usar 1 fila basta para el ancho)
                    Xtr_sample = pre_fitted.transform(X.iloc[:1])
                    n_final_cols = int(Xtr_sample.shape[1])
                    n_rows = int(len(X))

                    # Info de SelectKBest si existe (k solicitado y n_features previos)
                    sel = pre_fitted.named_steps.get("selectk", None) if hasattr(pre_fitted, "named_steps") else None
                    if sel is not None:
                        k_req = getattr(sel, "k", None)
                        n_before = None
                        try:
                            n_before = len(getattr(sel, "scores_", []))  # #features evaluadas por k-best
                        except Exception:
                            pass
                        logger.info(
                            f"Preprocessed shape: ({n_rows}, {n_final_cols}) | "
                            f"SelectKBest(k={k_req}, before={n_before}, selected={n_final_cols})"
                        )
                    else:
                        logger.info(f"Preprocessed shape: ({n_rows}, {n_final_cols}) (sin SelectKBest)")
                except Exception as _e:
                    logger.warning(f"No se pudo loguear shape preprocesado: {_e}")

            else:
                logger.warning("No se pudo exportar el dataset preprocesado (CSV).")
        else:
            logger.warning("Final estimator no expone step 'pre'; se omite exportación del dataset preprocesado.")
    except Exception as e:
        logger.warning(f"Fallo al exportar dataset preprocesado: {e}")

    # 11) Reporte final (+ métricas del final si se pospusieron)
    final_metrics: Dict[str, Any] = {}
    if ONLY_COMPUTE_METRICS_FOR_FINAL:
        if task == "classification":
            try:
                y_pred = cross_val_predict(final_est, X, y, cv=cv, n_jobs=n_jobs, method="predict", fit_params={})
            except TypeError:
                y_pred = cross_val_predict(final_est, X, y, cv=cv, n_jobs=n_jobs, method="predict")
            y_proba = None
            try:
                y_proba = cross_val_predict(final_est, X, y, cv=cv, n_jobs=n_jobs, method="predict_proba",
                                            fit_params={})
            except TypeError:
                try:
                    y_proba = cross_val_predict(final_est, X, y, cv=cv, n_jobs=n_jobs, method="predict_proba")
                except Exception:
                    y_proba = None
            final_metrics = evaluate_classification(y, y_pred, y_proba)
            if y_proba is not None and (y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 2)):
                thr, f1b = best_threshold_by_f1(y, y_proba)
                final_metrics["best_threshold_f1"] = thr
                final_metrics["best_threshold_f1_score"] = f1b
            if top_models:
                top_models[0]["metrics"] = final_metrics
        else:
            try:
                y_pred = cross_val_predict(final_est, X, y, cv=cv, n_jobs=n_jobs, method="predict", fit_params={})
            except TypeError:
                y_pred = cross_val_predict(final_est, X, y, cv=cv, n_jobs=n_jobs, method="predict")
            final_metrics = evaluate_regression(y, y_pred)
            if top_models:
                top_models[0]["metrics"] = final_metrics

    report = {
        "task": task,
        "best_models": top_models,
        "run_dir": run_dir,
        # labels_info si lo hubiera; por ahora exponemos mapping (cuando aplica)
        "labels_info": {"target_mapping": target_mapping} if target_mapping else None
    }
    _save_json(Path(run_dir) / report_name, report)

    # 12) Gráficas y OOF (clasificación)
    if task == "classification":
        try:
            plot_est = final_est
            try:
                y_pred = cross_val_predict(plot_est, X, y, cv=cv, n_jobs=n_jobs, method="predict", fit_params={})
            except TypeError:
                y_pred = cross_val_predict(plot_est, X, y, cv=cv, n_jobs=n_jobs, method="predict")
            try:
                y_proba = cross_val_predict(plot_est, X, y, cv=cv, n_jobs=n_jobs, method="predict_proba", fit_params={})
            except TypeError:
                try:
                    y_proba = cross_val_predict(plot_est, X, y, cv=cv, n_jobs=n_jobs, method="predict_proba")
                except Exception:
                    y_proba = None

            if MAKE_PLOTS:
                try:
                    plot_confusion(y, y_pred, str(Path(run_dir) / "confusion_matrix.png"))
                except Exception as e:
                    logger.warning(f"No se pudo guardar confusion_matrix.png: {e}")
                if y_proba is not None:
                    try:
                        plot_roc_curve(y, y_proba, str(Path(run_dir) / "roc_curve.png"))
                        plot_pr_curve(y, y_proba, str(Path(run_dir) / "pr_curve.png"))
                    except Exception as e:
                        logger.warning(f"No se pudo guardar ROC/PR: {e}")

            if SAVE_OOF_PREDICTIONS:
                try:
                    out = Path(run_dir) / "oof_predictions.csv"
                    if y_proba is not None:
                        if y_proba.ndim == 1:  # binario (proba clase 1)
                            df_oof = pd.DataFrame({
                                "y_true": y,
                                "y_pred": y_pred,
                                "proba_1": y_proba
                            })
                        else:
                            cols = [f"proba_{i}" for i in range(y_proba.shape[1])]
                            df_oof = pd.DataFrame({"y_true": y, "y_pred": y_pred})
                            for i, c in enumerate(cols):
                                df_oof[c] = y_proba[:, i]
                    else:
                        df_oof = pd.DataFrame({"y_true": y, "y_pred": y_pred})
                    df_oof.to_csv(out, index=False)
                except Exception as e:
                    logger.warning(f"No se pudo guardar oof_predictions.csv: {e}")
        except Exception as e:
            logger.warning(f"Failed plotting/evaluating final metrics: {e}")

    # 13) Importancias de características (si aplica)
    try:
        pre = None
        if isinstance(final_est, Pipeline):
            pre = final_est.named_steps.get("pre", None)
        df_imp = _try_feature_importance(final_est, pre, input_cols=list(X.columns))
        if df_imp is not None:
            df_imp.to_csv(Path(run_dir) / TOP_FEATURES_CSV, index=False)
    except Exception:
        pass

    # 14) README corto
    if getattr(cfg.train, "save_readme", False):
        with open(Path(run_dir) / "README.txt", "w", encoding="utf-8") as f:
            f.write(
                f"Run directory: {run_dir}\n"
                f"Task: {task}\n"
                f"Top models (by CV {primary_metric}):\n"
                f"{json.dumps(top_models, indent=2)}\n"
                f"Artifacts:\n"
                f"  - {pipe_name}\n"
                f"  - {report_name}\n"
                f"  - corr_heatmap.png, missing_bar.png\n"
                f"  - confusion_matrix.png / roc_curve.png / pr_curve.png\n"
                f"  - oof_predictions.csv, {TOP_FEATURES_CSV}\n"
                f"  - train_preprocessed.csv\n"
                f"  - target_mapping.json (si aplica)\n"
                f"  - auto_plan.json\n"
                f"  - env.json\n"
            )

    # 15) Enriquecer reporte con métricas de calibración/umbrales (no intrusivo)
    try:
        if task == 'classification':
            _enrich_report_with_calibration_and_thresholds(final_est, X, y, Path(run_dir))
    except Exception as e:
        logger.warning(f'No se pudo enriquecer reporte: {e}')

    # 16) Limpieza de caché de joblib (reduce warnings de resource_tracker)
    try:
        if pipeline_memory is not None:
            pipeline_memory.clear(warn=False)
    except Exception:
        pass

    return report

# python botPredictor.py --csv .\datasets\titanic\Titanic-Dataset.csv --target Survived --folds 5 --search-iters 30 --no-gpu
