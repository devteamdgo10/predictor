# trainer/utils.py
from __future__ import annotations

import os
import re
import json
import logging
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple, Iterable, Union

import numpy as np
import pandas as pd

# =========================
# Logger consistente
# =========================
LOG_FMT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
logger = logging.getLogger("trainer")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(LOG_FMT))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False


# =========================
# Utilidades base (compat)
# =========================
def set_seed(seed: int = 42) -> None:
    """Fija semillas para reproducibilidad (Python, NumPy y, si existe, TF)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> str:
    """Crea el directorio si no existe y devuelve el path."""
    os.makedirs(path, exist_ok=True)
    return path


def timestamped_dir(base: str, prefix: str = "run") -> str:
    """Devuelve un subdirectorio con timestamp dentro de base."""
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, f"{prefix}_{ts}")
    ensure_dir(out)
    return out


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Guarda JSON con indentación y UTF-8."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_csv(path: str, sample_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Carga CSV de forma segura. Si sample_rows está definido y es menor al total,
    devuelve una muestra aleatoria (semilla fija).
    """
    # Mantener comportamiento simple/estable
    df = pd.read_csv(path)
    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(sample_rows, random_state=42).reset_index(drop=True)
    return df


def detect_problem_type(y: pd.Series, override: Optional[str] = None) -> str:
    """
    Heurística segura para tarea: 'classification' si baja cardinalidad o dtype no numérico.
    Respeta override si es válido.
    """
    if override in ("classification", "regression"):
        return override
    nunique = y.nunique(dropna=True)
    if y.dtype == "object" or str(y.dtype).startswith(("category", "bool")):
        return "classification"
    if np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.floating):
        if nunique <= max(20, int(0.05 * len(y))):
            return "classification"
        return "regression"
    return "classification"


def guess_id_columns(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """
    Detecta columnas tipo ID según patrones y razón de unicidad (>90% distintos).
    No elimina; sólo sugiere.
    """
    cols: List[str] = []
    if not patterns:
        return cols
    try:
        regex = re.compile("|".join([re.escape(p) for p in patterns]), re.IGNORECASE)
    except re.error:
        # Si los patrones no son válidos, no sugerimos nada
        return cols

    for c in df.columns:
        if regex.search(c):
            uniq_ratio = df[c].nunique(dropna=True) / max(1, len(df))
            if uniq_ratio > 0.9:
                cols.append(c)
    return cols


# =========================
# Helpers nuevos (no intrusivos)
# =========================
def collect_env(extra_modules: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Captura versiones de entorno relevantes (Python, numpy, pandas, sklearn, xgboost, lightgbm, scikeras, tensorflow).
    No falla si algún paquete no está instalado.
    """
    info: Dict[str, Any] = {}
    try:
        import sys, sklearn, numpy, pandas  # type: ignore
        info.update({
            "python": sys.version,
            "numpy": getattr(numpy, "__version__", None),
            "pandas": getattr(pandas, "__version__", None),
            "sklearn": getattr(sklearn, "__version__", None),
        })
    except Exception:
        pass

    for modname in ["xgboost", "lightgbm", "scikeras", "tensorflow"]:
        try:
            mod = __import__(modname)
            info[modname] = getattr(mod, "__version__", None)
        except Exception:
            info[modname] = None

    if extra_modules:
        for modname in extra_modules:
            try:
                mod = __import__(modname)
                info[modname] = getattr(mod, "__version__", None)
            except Exception:
                info[modname] = None
    return info


def detect_date_columns(df: pd.DataFrame, sample: int = 200) -> List[str]:
    """
    Heurística suave para detectar columnas fecha:
      - dtype datetime64
      - o parseables (en una muestra) con errores coercitivos.
    """
    dates: List[str] = []
    for c in df.columns:
        s = df[c]
        if np.issubdtype(s.dtype, np.datetime64):
            dates.append(c)
            continue
        if s.dtype == object:
            try:
                sm = s.dropna().astype(str).head(sample)
                parsed = pd.to_datetime(sm, errors="coerce", dayfirst=False, infer_datetime_format=True)
                if parsed.notna().mean() > 0.8:
                    dates.append(c)
            except Exception:
                continue
    return dates


def infer_positive_label(y: Iterable[Any]) -> Any:
    """
    Determina la etiqueta positiva en un problema binario con heurística:
    preferidos=[1,'1',True,'Y','Yes','Si','S','True'] si está presente; si no, usa la minoritaria.
    """
    s = pd.Series(list(y)).dropna()
    uniq = s.unique()
    if len(uniq) != 2:
        # No binario; sin inferencia
        return None
    preferred = [1, "1", True, "Y", "y", "Yes", "YES", "Si", "SI", "S", "True", "TRUE"]
    for cand in preferred:
        if cand in uniq:
            return cand
    # minoritaria
    vc = s.value_counts()
    return vc.idxmin()


def normalize_binary_labels(y: Iterable[Any]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Mapea etiquetas binarias {neg,pos} -> {0,1} manteniendo NA.
    Devuelve (y_mapeado, mapping_str->int).
    """
    s = pd.Series(list(y))
    uniq = s.dropna().unique()
    if len(uniq) != 2:
        raise ValueError("normalize_binary_labels requiere 2 clases.")
    pos = infer_positive_label(s)
    if pos is None:
        # fallback: ordenar alfabéticamente
        uniq_sorted = sorted(uniq, key=lambda v: str(v))
        pos = uniq_sorted[1]
    neg = uniq[0] if uniq[0] != pos else uniq[1]
    mapping = {str(neg): 0, str(pos): 1}
    y_num = s.map(lambda v: mapping.get(str(v)) if pd.notna(v) else np.nan).to_numpy()
    return y_num, mapping


def boolish_to_bool(x: Any) -> Optional[bool]:
    """
    Convierte valores tipo 'Y/N', 'Yes/No', 'true/false', 1/0 a booleano. NA si no reconocido.
    """
    if pd.isna(x):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"y", "yes", "si", "s", "true", "1"}:
        return True
    if s in {"n", "no", "false", "0"}:
        return False
    return None


def compute_rare_categories(series: pd.Series,
                            min_freq: float = 0.01,
                            min_count: Optional[int] = None) -> List[Any]:
    """
    Devuelve la lista de categorías raras en 'series' cuyo soporte es < min_freq (o < min_count si se define).
    """
    vc = series.value_counts(dropna=False)
    n = len(series)
    rare: List[Any] = []
    for val, cnt in vc.items():
        freq = cnt / max(1, n)
        if (min_count is not None and cnt < min_count) or (min_count is None and freq < min_freq):
            rare.append(val)
    return rare


def collapse_rare_categories(series: pd.Series,
                             min_freq: float = 0.01,
                             min_count: Optional[int] = None,
                             other_label: str = "__OTHER__") -> pd.Series:
    """
    Colapsa categorías con baja frecuencia a una etiqueta común (por defecto '__OTHER__').
    No altera NaNs.
    """
    rare = set(compute_rare_categories(series, min_freq=min_freq, min_count=min_count))
    if not rare:
        return series
    return series.map(lambda v: other_label if (v in rare and pd.notna(v)) else v)


def safe_json_dumps(obj: Any) -> str:
    """`json.dumps` tolerante (sin excepciones) → str; si falla, repr(obj)."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False, indent=2)
        except Exception:
            return repr(obj)


def dataframe_profile(df: pd.DataFrame, topn: int = 10) -> Dict[str, Any]:
    """
    Resumen ligero de un DataFrame (columnas, dtypes, nulos y top valores por columna).
    Útil para logging/debug; no intrusivo.
    """
    profile: Dict[str, Any] = {
        "shape": list(df.shape),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nulls_top": {},
        "top_values": {}
    }
    nulls = df.isna().sum().sort_values(ascending=False)
    profile["nulls_top"] = {c: int(n) for c, n in nulls.head(topn).items() if n > 0}
    for c in df.columns:
        try:
            vc = df[c].value_counts(dropna=True).head(topn)
            profile["top_values"][c] = {str(k): int(v) for k, v in vc.items()}
        except Exception:
            profile["top_values"][c] = {}
    return profile


def detect_group_like_columns(df: pd.DataFrame,
                              name_hints: Optional[List[str]] = None,
                              max_nunique_ratio: float = 0.5) -> List[str]:
    """
    Heurística para columnas 'agrupables' (segmento, grupo, categoría):
    - nombre sugiere grupo/segmento/categoría (opcional),
    - y cardinalidad moderada (<= max_nunique_ratio del dataset).
    """
    hints = name_hints or ["group", "segment", "categoria", "category", "cluster", "tipo", "type"]
    res: List[str] = []
    n = len(df)
    for c in df.columns:
        nunq = df[c].nunique(dropna=True)
        if n <= 0:
            continue
        if nunq / n <= max_nunique_ratio:
            lowname = c.lower()
            if any(h in lowname for h in hints):
                res.append(c)
    return res
