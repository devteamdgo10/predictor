# trainer/auto_decider.py
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Usa el mismo logger que el resto del sistema
logger = logging.getLogger("trainer")


# ----------------------------
# Utilitarios internos
# ----------------------------
def _value_counts_safe(s: pd.Series) -> pd.Series:
    try:
        return s.value_counts(dropna=False)
    except Exception:
        return pd.Series(dtype="int64")


def _class_imbalance_info(y: pd.Series) -> Tuple[float, float]:
    """(ratio_max/min, positive_rate aprox para binario)"""
    vc = _value_counts_safe(pd.Series(y))
    if len(vc) < 2:
        return 1.0, 1.0
    ratio = float(vc.max()) / float(vc.min())
    # Para binario intentamos usar la última clave como "positiva" (estable con pandas)
    pos_rate = float((y == vc.index[-1]).sum()) / len(y) if len(vc) == 2 else float(vc.max()) / len(y)
    return ratio, pos_rate


def _skewed_numeric_columns(X: pd.DataFrame, skew_thresh: float = 1.0) -> List[str]:
    num = X.select_dtypes(include=[np.number])
    if num.empty:
        return []
    with np.errstate(all="ignore"):
        sk = num.skew(numeric_only=True)
    return [c for c, v in sk.items() if np.isfinite(v) and abs(v) >= skew_thresh]


def _high_card_cols(X: pd.DataFrame, threshold: int = 50) -> List[str]:
    cats = X.select_dtypes(exclude=[np.number])
    cols: List[str] = []
    for c in cats.columns:
        try:
            nun = X[c].nunique(dropna=False)
            if nun >= threshold:
                cols.append(c)
        except Exception:
            continue
    return cols


def _auto_k_best(n_features: int) -> int:
    """k proporcional al ancho con piso/techo suaves."""
    if n_features <= 0:
        return 10
    return min(max(10, int(round(n_features * 0.6))), 80)


def _suggest_search_strategy(n_samples: int) -> str:
    """
    Heurística ligera:
      - >= 10k → 'halving'
      - [2k, 10k) → 'random' (pero con iters extra por boosting)
      - < 2k → 'random'
    """
    if n_samples >= 10_000:
        return "halving"
    return "random"


def _distribute_iters(models: List[str], total_budget: int, n_samples: int) -> Dict[str, int]:
    """
    Reparte presupuesto de búsqueda por modelo.
    Favorece boosting en datasets medianos/grandes.
    """
    if total_budget is None or total_budget <= 0:
        total_budget = 30

    # Pesos base por familia
    base_w = {
        "logistic": 1.0,
        "rf": 1.2,
        "xgb": 1.8,
        "lgbm": 1.8,
        "mlp": 1.4,
        "linear": 1.0,
        "catboost": 1.8,
    }

    # Ajuste simple por tamaño
    if n_samples >= 5000:
        base_w["xgb"] += 0.4
        base_w["lgbm"] += 0.4

    weights = np.array([base_w.get(m, 1.0) for m in models], dtype=float)
    weights = np.maximum(weights, 0.1)
    weights = weights / weights.sum()

    # Asegurar al menos 3 por modelo
    iters = np.maximum((weights * total_budget).round().astype(int), 3)
    # Reajuste fino para cuadrar presupuesto
    diff = int(total_budget - iters.sum())
    for i in range(abs(diff)):
        idx = i % len(iters)
        iters[idx] += 1 if diff > 0 else -1
        iters[idx] = max(iters[idx], 1)

    return {m: int(n) for m, n in zip(models, iters.tolist())}


def _suggest_resample(task: str, ratio: float, n_samples: int) -> bool:
    """
    Activar re-muestreo (SMOTE) sólo cuando:
      - Es clasificación
      - Desbalance notable (ratio ≥ 2.0)
      - Dataset pequeño/mediano (≤ 50k) para mantener tiempos razonables
    """
    if task != "classification":
        return False
    if ratio < 2.0:
        return False
    if n_samples > 50_000:
        return False
    return True


# ----------------------------
# Plan y API pública
# ----------------------------
@dataclass
class AutoPlan:
    task: str
    primary_metric: str
    class_weight: Optional[str]
    do_calibration: bool

    preprocess: Dict[str, Any]
    model_names: List[str]
    model_iters: Dict[str, int]

    dataset_profile: Dict[str, Any]
    labels_info: Optional[Dict[str, Any]] = None

    search_strategy: str = "random"
    early_stopping: bool = True
    enable_cache: bool = True  # permite caching de pipeline si el caller lo usa

    # NUEVO: sugerencia de re-muestreo para manejar desbalance dentro del CV
    resample: bool = False

    rationale: Dict[str, str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Limpieza: no exportar None innecesarios
        if d.get("class_weight") is None:
            d["class_weight"] = None
        if d.get("labels_info") is None:
            d["labels_info"] = None
        return d


def build_auto_plan(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: Optional[str] = None,
    cfg: Optional[Any] = None,
    candidate_models: Optional[List[str]] = None,
    total_search_iters: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Genera un plan de entrenamiento 'best-effort' a partir del dataset.
    - No modifica X ni y.
    - Retorna un dict sencillo; el caller decide qué partes usar.
    """
    # --- Perfil del dataset
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    ratio, pos_rate = _class_imbalance_info(y)
    skewed = _skewed_numeric_columns(X, skew_thresh=1.0)

    # Umbral de alta cardinalidad desde cfg si existe
    high_card_threshold = getattr(getattr(cfg, "preprocess", None), "high_card_threshold", 50) or 50
    high_card = _high_card_cols(X, threshold=int(high_card_threshold))

    inferred_task = task or _infer_task_from_y(y)

    dataset_profile = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_num": len(num_cols),
        "n_cat": len(cat_cols),
        "skewed_num_cols_sample": skewed[:10],  # muestra
        "high_card_cat_cols": high_card,
        "imbalance_ratio": float(ratio),
    }

    logger.info(
        "Auto-plan | Perfil dataset: n=%s, d=%s (num=%s, cat=%s), desbalance≈%.2f, high_card=%s",
        n_samples, n_features, len(num_cols), len(cat_cols), ratio, high_card[:5],
    )

    # --- Métrica primaria
    primary_metric_override = getattr(getattr(cfg, "train", None), "primary_metric", None)
    if primary_metric_override:
        primary_metric = primary_metric_override
        logger.info("Auto-plan | Métrica primaria definida por config: %s", primary_metric)
    else:
        primary_metric = (
            "f1_macro" if inferred_task == "classification" and ratio >= 1.5
            else "roc_auc" if inferred_task == "classification"
            else "neg_root_mean_squared_error"
        )
        logger.info("Auto-plan | Métrica primaria sugerida: %s", primary_metric)

    # --- Preprocesamiento sugerido (opt-in)
    select_k_best_cfg = getattr(getattr(cfg, "preprocess", None), "select_k_best", None)
    select_k_best = _auto_k_best(n_features) if (select_k_best_cfg in (None, 0, "auto")) else select_k_best_cfg

    power_transform_cfg = getattr(getattr(cfg, "preprocess", None), "power_transform", None)
    power_transform = True if (power_transform_cfg in (None, "auto") and len(skewed) > 0) else bool(power_transform_cfg)

    scale_for_cfg = getattr(getattr(cfg, "preprocess", None), "scale_for", None)
    scale_for = scale_for_cfg or ["linear", "logistic", "mlp"]

    one_hot_for_cfg = getattr(getattr(cfg, "preprocess", None), "one_hot_for", None)
    one_hot_for = one_hot_for_cfg  # mantenemos lo que ya tengas

    drop_low_variance = getattr(getattr(cfg, "preprocess", None), "drop_low_variance", False)
    low_variance_threshold = getattr(getattr(cfg, "preprocess", None), "low_variance_threshold", 0.0)

    freq_encode_high_card_cfg = getattr(getattr(cfg, "preprocess", None), "freq_encode_high_card", True)
    high_card_threshold_cfg = getattr(getattr(cfg, "preprocess", None), "high_card_threshold", high_card_threshold)

    preprocess_plan = {
        "scale_for": scale_for,
        "power_transform": bool(power_transform),
        "select_k_best": int(select_k_best) if select_k_best else None,
        "one_hot_for": one_hot_for,
        "freq_encode_high_card": bool(freq_encode_high_card_cfg),
        "high_card_threshold": int(high_card_threshold_cfg),
        "drop_low_variance": bool(drop_low_variance),
        "low_variance_threshold": float(low_variance_threshold),
        "numeric_imputer": getattr(getattr(cfg, "preprocess", None), "numeric_imputer", "median"),
        "categorical_imputer": getattr(getattr(cfg, "preprocess", None), "categorical_imputer", "most_frequent"),
    }

    logger.info(
        "Auto-plan | Preprocess: k_best=%s, power_transform=%s, scale_for=%s, "
        "freq_encode=%s(th=%s), drop_low_var=%s",
        preprocess_plan["select_k_best"],
        preprocess_plan["power_transform"],
        preprocess_plan["scale_for"],
        preprocess_plan["freq_encode_high_card"],
        preprocess_plan["high_card_threshold"],
        preprocess_plan["drop_low_variance"],
    )

    # --- Class weight / calibración
    class_weight_cfg = getattr(getattr(cfg, "train", None), "class_weight", None)
    class_weight = class_weight_cfg
    if inferred_task == "classification" and class_weight_cfg is None and ratio >= 1.5:
        class_weight = "balanced"
        logger.info("Auto-plan | class_weight='balanced' sugerido por desbalance alto (≈%.2f).", ratio)

    do_calibration_cfg = getattr(getattr(cfg, "train", None), "calibrate", None)
    do_calibration = bool(do_calibration_cfg) if do_calibration_cfg is not None else False

    # --- Modelos candidatos
    if candidate_models is None:
        candidate_models = ["logistic", "rf", "xgb", "lgbm"]

    # --- Presupuesto de búsqueda
    total_iters = (
        getattr(getattr(cfg, "cv", None), "n_iter_search", None)
        or getattr(getattr(cfg, "cv", None), "n_iter", None)
        or total_search_iters
        or 30
    )
    # Nota: mantenemos el comportamiento previo (repartir sobre total_iters * n_modelos)
    model_iters = _distribute_iters(candidate_models, int(total_iters) * len(candidate_models), n_samples)

    # --- Estrategia de búsqueda y early stopping
    search_strategy_cfg = getattr(getattr(cfg, "train", None), "search_strategy", None)
    search_strategy = search_strategy_cfg or _suggest_search_strategy(n_samples)
    early_stopping_cfg = getattr(getattr(cfg, "train", None), "early_stopping", None)
    early_stopping = True if early_stopping_cfg is None else bool(early_stopping_cfg)

    # --- Cache de pipeline
    enable_cache_cfg = getattr(getattr(cfg, "train", None), "enable_cache", None)
    enable_cache = True if enable_cache_cfg is None else bool(enable_cache_cfg)

    # --- NUEVO: sugerencia de re-muestreo (SMOTE) para CV
    resample = _suggest_resample(inferred_task, ratio, n_samples)

    # Log del presupuesto con suma real
    sum_iters = int(sum(model_iters.values()))
    logger.info(
        "Auto-plan | Strategy: %s | EarlyStopping=%s | Cache=%s | Budget total≈%s → %s (suma≈%s)",
        search_strategy, early_stopping, enable_cache, total_iters, model_iters, sum_iters
    )

    rationale = {
        "metric": "f1_macro si desbalance ≥1.5; si no, roc_auc (clasif) / RMSE (reg).",
        "calibration": "activar con desbalance fuerte si se requieren probas calibradas.",
        "power_transform": "activar si skew≥1.0 en numéricas.",
        "select_k_best": "k≈0.6*ancho con límites [10,80].",
        "freq_encode": f"usar cuando cat con cardinalidad ≥{preprocess_plan['high_card_threshold']}.",
        "iters": "prioriza boosting en medianos/grandes; reparto proporcional por familia.",
        "search": "halving en datasets grandes; random para el resto (con fallback).",
        "resample": "SMOTE dentro de CV si clasif. con ratio≥2.0 y n_samples≤50k.",
    }

    plan = AutoPlan(
        task=inferred_task,
        primary_metric=primary_metric,
        class_weight=class_weight,
        do_calibration=do_calibration,
        preprocess=preprocess_plan,
        model_names=candidate_models,
        model_iters=model_iters,
        dataset_profile=dataset_profile,
        labels_info=None,
        search_strategy=search_strategy,
        early_stopping=early_stopping,
        enable_cache=enable_cache,
        resample=resample,
        rationale=rationale,
    )

    return plan.to_dict()


def _infer_task_from_y(y: pd.Series) -> str:
    y = pd.Series(y).dropna()
    # Si es numérico con muchos valores distintos → regresión
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique(dropna=True) > 20:
            return "regression"
        else:
            return "classification"
    # Si es objeto/categórico → si <=20 clases, clasificación
    try:
        if y.nunique(dropna=True) <= 20:
            return "classification"
    except Exception:
        pass
    return "regression"
