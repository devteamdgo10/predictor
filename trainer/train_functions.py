# trainer/train_functions.py
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import keras
from typing import Optional

from scikeras.wrappers import KerasClassifier, KerasRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Mantén el mismo nombre del step del estimador en el Pipeline
MODEL_STEP = "model"


# =========================
# Utilitarios locales
# =========================
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _timestamped_dir(base: str | Path, prefix: str = "run") -> str:
    from datetime import datetime
    base = Path(base)
    _ensure_dir(base)
    run_dir = base / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _ensure_dir(run_dir)
    return str(run_dir)


def _save_json(path: str | Path, data: Dict[str, Any]) -> None:
    import json
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass


def _detect_problem_type(y: pd.Series, override: Optional[str] = None) -> str:
    if override in {"classification", "regression"}:
        return override
    y_unique = pd.Series(y).dropna().unique()
    # Si la cardinalidad es baja, asumimos clasificación (aunque sea 'object').
    return "classification" if len(y_unique) <= 20 else "regression"


def _class_imbalance_info(y: pd.Series) -> Tuple[float, float]:
    """Devuelve (ratio_mayoría/minoría, positive_rate). Safe para binario y multiclase."""
    vc = pd.Series(y).value_counts(dropna=False)
    if len(vc) < 2:
        return 1.0, float(vc.iloc[0]) / len(y)
    ratio = float(vc.max()) / float(vc.min())
    positive_rate = float((y == vc.index[-1]).sum()) / len(y) if len(vc) == 2 else float(vc.max()) / len(y)
    return ratio, positive_rate


def _maybe_add_missing_indicators(X: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    miss = X.isna().sum()
    cols = [c for c, n in miss.items() if n > 0]
    for c in cols:
        X[f"MI__{c}"] = X[c].isna().astype(int)
    return X, len(cols)


def _auto_select_k(num_cols: int) -> Optional[int]:
    """Heurística suave: k proporcional al ancho, con top y floor."""
    if num_cols <= 0:
        return None
    k = min(max(10, int(num_cols * 0.6)), 80)  # entre 10 y 80 características
    return k


def _skewed_numeric_columns(X: pd.DataFrame, skew_thresh: float = 1.0) -> List[str]:
    num = X.select_dtypes(include=[np.number]).copy()
    if num.empty:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = num.skew(numeric_only=True)
    return [c for c, v in sk.items() if np.isfinite(v) and abs(v) >= skew_thresh]


def _save_corr_heatmap(X: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    num = X.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return
    corr = num.corr().astype(float)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_missing_bar(X: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    miss = X.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.bar(miss.index, miss.values)
    ax.set_ylabel("Valores nulos")
    # FIX mínimo del warning: definir ticks antes de set_xticklabels
    ax.set_xticks(range(len(miss.index)))
    ax.set_xticklabels(miss.index, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _try_feature_importance(final_est, pre, input_cols: List[str]) -> Optional[pd.DataFrame]:
    """Intenta mapear importancias/coefs a nombres de features transformados."""
    try:
        feat_names = None
        if hasattr(pre, "get_feature_names_out"):
            try:
                feat_names = list(pre.get_feature_names_out(input_features=input_cols))
            except Exception:
                feat_names = list(pre.get_feature_names_out())
        elif hasattr(pre, "get_feature_names"):
            feat_names = list(pre.get_feature_names())

        model = final_est
        # Si final_est es un Pipeline(pre, model)
        if isinstance(final_est, Pipeline):
            model = final_est.named_steps.get(MODEL_STEP, model)
            pre = final_est.named_steps.get("pre", pre)

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coefs = np.asarray(model.coef_, dtype=float)
            importances = np.mean(np.abs(coefs), axis=0) if coefs.ndim > 1 else np.abs(coefs)
        else:
            return None

        if feat_names is None or len(feat_names) != len(importances):
            # fallback a índices
            feat_names = [f"f_{i}" for i in range(len(importances))]

        df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
        df_imp.sort_values("importance", ascending=False, inplace=True)
        return df_imp
    except Exception:
        return None


# =========================
# Helpers de CV y métrica
# =========================
def _cv_for_task(task: str, folds: int, shuffle: bool, random_state: int):
    from sklearn.model_selection import StratifiedKFold, KFold
    return StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state) if task == "classification" \
        else KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)


def _primary_metric_default(task: str) -> str:
    return "roc_auc" if task == "classification" else "neg_root_mean_squared_error"


def _candidate_models(task: str, include_mlp: bool = False) -> List[str]:
    if task == "classification":
        names = ["logistic", "rf", "xgb", "lgbm"]
    else:
        names = ["linear", "rf", "xgb", "lgbm"]
    if include_mlp:
        names.append("mlp")
    return names


# ==================================family ========
# Randomized/Halving Search con fallback a defaults
# ==========================================
def _fit_search(
        pipe,
        param_distributions: Optional[Dict[str, Any]],
        X,
        y,
        scorer: str,
        cv,
        n_iter: int,
        n_jobs: int,
        random_state: int,
        fit_params: Dict[str, Any],
        *,
        search_strategy: Optional[str] = None  # opcional: "halving" o None
):
    """Soporta RandomizedSearchCV y, si se pide y está disponible, HalvingRandomSearchCV.
       Durante la búsqueda se desactiva el cache; el modelo final (producción) se reentrena con el pipe original (y su memory)."""
    import numpy as _np
    import warnings as _warnings
    from sklearn.base import clone  # <-- para clonar pipes

    # === Detección de familia para ajustes de paralelización ===
    try:
        est = pipe.named_steps.get(MODEL_STEP, None)
        est_name = type(est).__name__ if est is not None else ""
        est_name_u = est_name.upper()
        _is_xgb = ("XGB" in est_name_u)  # XGBClassifier / XGBRegressor
        _is_lgbm = ("LGBM" in est_name_u)  # LGBMClassifier / LGBMRegressor
    except Exception:
        _is_xgb, _is_lgbm = False, False

    # Usa backend por hilos para XGB/LGBM (evita procesos + hilos internos)
    _backend = "threading" if (_is_xgb or _is_lgbm) else "loky"

    # Importes locales (con fallbacks seguros)
    try:
        from joblib import parallel_backend as _parallel_backend
    except Exception:
        _parallel_backend = None
    try:
        from threadpoolctl import threadpool_limits as _threadpool_limits
    except Exception:
        _threadpool_limits = None
    from contextlib import nullcontext as _nullcontext

    # Contextos
    _limit_ctx = _threadpool_limits(limits=1) if _threadpool_limits else _nullcontext()
    _pb_ctx = _parallel_backend(_backend) if _parallel_backend else _nullcontext()

    use_halving = (search_strategy == "halving")

    # Normaliza el espacio (prefijo del step del estimador)
    if param_distributions:
        param_distributions = {f"{MODEL_STEP}__{k}": v for k, v in param_distributions.items()}

    # Helper para CV simple con defaults del estimador (mantiene cache de tu pipe)
    def _fit_defaults_and_score():
        with _limit_ctx, _pb_ctx:
            est_prod = pipe.fit(X, y, **fit_params)
            score = float(_np.mean(cross_val_score(est_prod, X, y, cv=cv, scoring=scorer, n_jobs=n_jobs)))
            return est_prod, est_prod, score

    if not param_distributions:
        # Sin espacio de búsqueda → fit directo con cache (producción)
        return _fit_defaults_and_score()

    # Calcula n_iter efectivo (por si la grid es pequeña)
    try:
        space_size = sum(len(v) if isinstance(v, list) else 1 for v in param_distributions.values())
        n_iter_eff = min(n_iter, max(1, space_size))
    except Exception:
        n_iter_eff = n_iter

    # Cap suave para XGB/LGBM
    if (_is_xgb or _is_lgbm) and n_iter_eff > 12:
        n_iter_eff = 12

    # --- APAGAR CACHE SOLO PARA LA BÚSQUEDA ---
    pipe_search = clone(pipe)
    try:
        pipe_search.set_params(memory=None)
    except Exception:
        pass
    # ------------------------------------------

    # 1) Intento con HALVING si procede (refit=False para que el refit final lo hagamos con el pipe original y su memory)
    if use_halving:
        try:
            from sklearn.experimental import enable_halving_search_cv  # noqa: F401
            from sklearn.model_selection import HalvingRandomSearchCV
            search = HalvingRandomSearchCV(
                estimator=pipe_search,
                param_distributions=param_distributions,
                factor=3,
                resource="n_samples",
                max_resources="auto",
                scoring=scorer,
                cv=cv,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=2,
                refit=False,  # <<< búsqueda sin refit (sin cache)
                error_score="raise"
            )
            with _limit_ctx, _pb_ctx:
                fitted_search = search.fit(X, y, **fit_params)

            best_params = search.best_params_
            # Reentrena en PRODUCCIÓN con tu pipe original (cache ON)
            final_pipe = clone(pipe)
            try:
                final_pipe.set_params(**best_params)  # acepta "model__param"
            except Exception:
                # Intento más fino: aplicar solo al step 'model'
                try:
                    clean_params = {k.split(f"{MODEL_STEP}__")[-1]: v
                                    for k, v in best_params.items()
                                    if k.startswith(f"{MODEL_STEP}__")}
                    final_pipe.named_steps[MODEL_STEP].set_params(**clean_params)
                except Exception:
                    pass

            with _limit_ctx, _pb_ctx:
                final_fitted = final_pipe.fit(X, y, **fit_params)

            return fitted_search, final_fitted, float(search.best_score_)
        except Exception as e:
            _warnings.warn(
                f"HalvingRandomSearchCV no disponible o falló ({e}). "
                f"Se usa RandomizedSearchCV como fallback."
            )

    # 2) RandomizedSearchCV (ruta estándar) — también refit=False y reentreno final con cache
    try:
        search = RandomizedSearchCV(
            estimator=pipe_search,
            param_distributions=param_distributions,
            n_iter=n_iter_eff,
            scoring=scorer,
            cv=cv,
            refit=False,  # <<< búsqueda sin refit (sin cache)
            verbose=2,
            n_jobs=n_jobs,
            random_state=random_state,
            pre_dispatch="2*n_jobs",
            error_score="raise"
        )
        with _limit_ctx, _pb_ctx:
            fitted_search = search.fit(X, y, **fit_params)

        best_params = search.best_params_

        # Reentrena en PRODUCCIÓN con tu pipe original (cache ON)
        final_pipe = clone(pipe)
        try:
            final_pipe.set_params(**best_params)  # acepta "model__param"
        except Exception:
            try:
                clean_params = {k.split(f"{MODEL_STEP}__")[-1]: v
                                for k, v in best_params.items()
                                if k.startswith(f"{MODEL_STEP}__")}
                final_pipe.named_steps[MODEL_STEP].set_params(**clean_params)
            except Exception:
                pass

        with _limit_ctx, _pb_ctx:
            final_fitted = final_pipe.fit(X, y, **fit_params)

        return fitted_search, final_fitted, float(search.best_score_)
    except Exception as e:
        _warnings.warn(
            f"RandomizedSearchCV failed on {type(pipe.named_steps[MODEL_STEP]).__name__}: {e}. "
            f"Fitting defaults instead."
        )
        return _fit_defaults_and_score()


# =========================
# MLP builders (no usados por defecto)
# =========================
def _build_mlp_classifier():

    def make_model(meta: dict, hidden_layers=(128,), dropout=0.1, learning_rate=1e-3):
        input_dim = int(meta["n_features_in_"])
        n_classes = int(meta.get("n_classes_", 2))
        target_type = meta.get("target_type_", "binary")
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for u in hidden_layers:
            x = keras.layers.Dense(u, activation="relu")(x)
            if dropout and dropout > 0:
                x = keras.layers.Dropout(dropout)(x)
        if target_type == "binary" or n_classes == 2:
            outputs = keras.layers.Dense(1, activation="sigmoid")(x)
            loss = "binary_crossentropy"
        else:
            outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
            loss = "sparse_categorical_crossentropy"
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)
        return model

    return KerasClassifier(
        model=make_model, hidden_layers=(128,), dropout=0.1, learning_rate=1e-3,
        epochs=25, batch_size=128, verbose=0
    )


def _build_mlp_regressor():
    def make_model(meta: dict, hidden_layers=(128,), dropout=0.1, learning_rate=1e-3):
        input_dim = int(meta["n_features_in_"])
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for u in hidden_layers:
            x = keras.layers.Dense(u, activation="relu")(x)
            if dropout and dropout > 0:
                x = keras.layers.Dropout(dropout)(x)
        outputs = keras.layers.Dense(1, activation="linear")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
        return model

    return KerasRegressor(
        model=make_model, hidden_layers=(128,), dropout=0.1, learning_rate=1e-3,
        epochs=25, batch_size=128, verbose=0
    )


# =========================
# Export del dataset preprocesado
# =========================
def export_preprocessed_dataset(fitted_pre, X: pd.DataFrame, y: pd.Series, out_path: Path) -> bool:
    """
    Exporta a CSV el dataset ya transformado por el preprocesador 'fitted_pre'.
    La primera columna es 'y'. Intenta usar nombres reales de features.
    """
    try:
        # Transform
        Xtr = fitted_pre.transform(X)

        # Nombres de columnas
        feat_names: Optional[List[str]] = None
        if hasattr(fitted_pre, "get_feature_names_out"):
            try:
                feat_names = list(fitted_pre.get_feature_names_out(input_features=X.columns))
            except Exception:
                try:
                    feat_names = list(fitted_pre.get_feature_names_out())
                except Exception:
                    feat_names = None

        # Densificar si es sparse
        try:
            from scipy import sparse
            if sparse.issparse(Xtr):
                Xtr = Xtr.toarray()
        except Exception:
            # Si no está scipy, intentamos toarray() si existe
            if hasattr(Xtr, "toarray"):
                Xtr = Xtr.toarray()

        # Asegurar dimensiones y nombres
        n_cols = Xtr.shape[1]
        if not feat_names or len(feat_names) != n_cols:
            feat_names = [f"f_{i}" for i in range(n_cols)]

        df_out = pd.DataFrame(Xtr, columns=feat_names)
        df_out.insert(0, "y", pd.Series(y).reset_index(drop=True))

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        return True
    except Exception:
        return False


# =========================
# Helper opcional: extraer 'pre' de un modelo final
# =========================
# trainer/train_functions.py


def get_preprocessor_from(final_estimator) -> Optional[Pipeline]:
    """
    Devuelve el step 'pre' aunque el final sea CalibratedClassifierCV u otros contenedores.
    Orden de preferencia:
      1) Si es Pipeline directo → 'pre'
      2) Atributos comunes que referencian el estimador (estimator/base_estimator[/_])
      3) Dentro de calibrated_classifiers_ (estimadores ya ajustados por fold)
    """
    est = final_estimator

    # 1) Pipeline directo
    if isinstance(est, Pipeline):
        return est.named_steps.get("pre", None)

    # 2) Atributos típicos (compat con distintas versiones)
    for attr in ("estimator", "estimator_", "base_estimator", "base_estimator_"):
        inner = getattr(est, attr, None)
        if isinstance(inner, Pipeline):
            return inner.named_steps.get("pre", None)

    # 3) CalibratedClassifierCV: buscar en los calibradores por fold
    c_list = getattr(est, "calibrated_classifiers_", None)
    if c_list:
        for c in c_list:
            for attr in ("estimator", "base_estimator", "classifier"):
                inner = getattr(c, attr, None)
                if isinstance(inner, Pipeline):
                    return inner.named_steps.get("pre", None)

    return None
