# trainer/features.py
from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, PowerTransformer,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import TruncatedSVD
from scipy import sparse


# ---- utilidades ----
def split_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    obj_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, obj_cols, X.columns.tolist()


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Mapea columnas categóricas ordinales según 'mappings'.
    REGLA sklearn: __init__ NO debe modificar los parámetros (deja mappings tal cual).
    """
    def __init__(self, mappings: Optional[Dict[str, Dict[str, int]]] = None):
        self.mappings = mappings  # <- NO tocar ni sustituir
        self.columns_: List[str] = []

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.mappings:
            for col, mp in self.mappings.items():
                if col in X.columns:
                    X[col] = X[col].map(mp)
        return X


class HighCardinalityFreqEncoder(BaseEstimator, TransformerMixin):
    """
    Reemplaza categóricas de alta cardinalidad por su frecuencia relativa.
    Respetando reglas sklearn:
      - __init__ NO modifica los parámetros recibidos.
      - Atributos aprendidos se guardan con sufijo '_' .
    """
    def __init__(self, threshold: int = 40, exclude_cols: Optional[List[str]] = None):
        self.threshold = threshold
        self.exclude_cols = exclude_cols  # <- NO normalizar aquí (para clone)

        # Atributos de fit
        self.freq_maps_: Dict[str, Dict[Union[str, float], float]] = {}
        self.high_card_cols_: List[str] = []
        self._exclude_set_ = None  # se define en fit

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        # Normaliza exclusiones en fit (no en __init__)
        self._exclude_set_ = set(self.exclude_cols) if self.exclude_cols else set()

        self.freq_maps_.clear()
        self.high_card_cols_.clear()

        for c in X.columns:
            if c in self._exclude_set_:
                continue
            dt = X[c].dtype
            if dt == object or str(dt).startswith("category"):
                try:
                    nunique = X[c].nunique(dropna=True)
                except Exception:
                    continue
                if nunique >= int(self.threshold):
                    vc = X[c].value_counts(normalize=True, dropna=False)
                    self.freq_maps_[c] = vc.to_dict()
                    self.high_card_cols_.append(c)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.high_card_cols_:
            # Map a frecuencia; NaN→0.0 como fallback estable
            X[c] = X[c].map(self.freq_maps_.get(c, {})).fillna(0.0).astype(float)
        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa categorías raras por columna: valores con frecuencia < min_freq
    pasan a '__OTHER__'. Si min_freq < 1, se interpreta como proporción.
    """
    def __init__(self, min_freq: float = 0.01):
        self.min_freq = float(min_freq)
        self.keep_values_: Dict[str, set] = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        n = len(X)
        for c in X.columns:
            if X[c].dtype == object or str(X[c].dtype).startswith("category"):
                vc = X[c].value_counts(dropna=False)
                if self.min_freq < 1.0:
                    keep = set(vc[vc >= self.min_freq * n].index)
                else:
                    keep = set(vc[vc >= int(self.min_freq)].index)
                self.keep_values_[c] = keep
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c, keep in self.keep_values_.items():
            if c in X.columns:
                X[c] = X[c].where(X[c].isin(keep), "__OTHER__")
        return X


class HashingTokenTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica hashing trick a columnas string detectadas (o provistas).
    Se generan tokens 'col=valor' por fila y se vectoriza con FeatureHasher.
    """
    def __init__(self, cols: Optional[List[str]] = None, n_features: int = 512, alternate_sign: bool = False):
        self.cols = cols
        self.n_features = int(n_features)
        self.alternate_sign = bool(alternate_sign)
        self.cols_: List[str] = []
        self.hasher_: Optional[FeatureHasher] = None

    def _auto_cols(self, X: pd.DataFrame) -> List[str]:
        cand: List[str] = []
        n = len(X)
        for c in X.columns:
            if X[c].dtype == object or str(X[c].dtype).startswith("category"):
                try:
                    nunique = X[c].nunique(dropna=True)
                    avg_len = X[c].astype(str).str.len().mean()
                    uniq_ratio = nunique / max(n, 1)
                    # heurística: texto largo o casi-ID (muchos únicos)
                    if (avg_len is not None and avg_len >= 20) or uniq_ratio >= 0.5:
                        cand.append(c)
                except Exception:
                    continue
        return cand

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.cols_ = list(self.cols) if self.cols is not None else self._auto_cols(X)
        self.hasher_ = FeatureHasher(
            n_features=self.n_features,
            input_type="string",
            alternate_sign=self.alternate_sign
        )
        # no necesita ajuste real
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if not self.cols_:
            # devuelve matriz vacía con 0 columnas
            return sparse.csr_matrix((len(X), 0))
        # construir lista de tokens por fila
        tokens_per_row = []
        for _, row in X[self.cols_].iterrows():
            toks = []
            for c in self.cols_:
                v = row[c]
                if pd.isna(v):
                    v = "__NA__"
                toks.append(f"{c}={str(v)}")
            tokens_per_row.append(toks)
        M = self.hasher_.transform(tokens_per_row)
        return M

    def get_feature_names_out(self, input_features=None):
        return np.array([f"hash_{i}" for i in range(self.n_features)], dtype=object)


class SVDWrapper(BaseEstimator, TransformerMixin):
    """Ajusta TruncatedSVD con n_components seguro (< n_features)."""
    def __init__(self, n_components: int = 256, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.svd_: Optional[TruncatedSVD] = None
        self._n_out: int = 0

    def fit(self, X, y=None):
        n_feats = X.shape[1]
        k = max(1, min(self.n_components, n_feats - 1)) if n_feats > 1 else 1
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.svd_.fit(X, y)
        self._n_out = k
        return self

    def transform(self, X):
        return self.svd_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"svd_{i}" for i in range(self._n_out)], dtype=object)


def _make_ohe_compat(want_sparse: bool = False):
    """
    Devuelve OneHotEncoder compatible con sklearn 1.3/1.4+:
    - 1.4+ usa `sparse_output=...`
    - <=1.3 usa `sparse=...`
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=want_sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=want_sparse)


def build_preprocessor(
    X: pd.DataFrame,
    family: str,
    select_k_best: int|None,
    scale_for: List[str],
    one_hot_for: List[str],
    numeric_imputer: str,
    categorical_imputer: str,
    drop_low_variance: bool,
    low_variance_threshold: float,
    task: str,
    ordinal_mappings: Optional[Dict[str, Dict[str,int]]],
    power_transform: bool,
    freq_encode_high_card: bool,
    high_card_threshold: int,
    # --- NUEVOS OPCIONALES (default seguros) ---
    rare_min_freq: Optional[float] = None,
    hash_text: bool = False,
    hash_n_features: int = 512,
    use_native_categoricals_for_lgbm: bool = False,   # sin OHE: OrdinalEncoder
    reduce_sparse: bool = False,
    svd_components: int = 256
) -> ColumnTransformer|Pipeline:

    num_cols, cat_cols, _ = split_feature_types(X)

    # Detectar columnas a hashear si aplica
    hashed_cols: List[str] = []
    if hash_text:
        # selección heurística automática; se excluyen del pipeline categórico
        ht = HashingTokenTransformer(cols=None, n_features=hash_n_features)
        hashed_cols = ht._auto_cols(pd.DataFrame(X))
        # no procesaremos esas columnas en cat_cols estándar
        cat_cols = [c for c in cat_cols if c not in hashed_cols]

    # ordinal primero (no altera parámetros)
    ord_mapper = OrdinalMapper(ordinal_mappings)

    # high cardinality freq encoding (evita columnas hashed)
    freq_enc = HighCardinalityFreqEncoder(
        threshold=high_card_threshold,
        exclude_cols=hashed_cols
    ) if freq_encode_high_card else None

    # === Ajuste mínimo: separar numéricos continuos vs binarios (especial MI__*) ===
    # binarios por prefijo MI__ o por cardinalidad ≤ 2
    bin_cols: List[str] = [c for c in X.columns if c.startswith('MI__')]
    for c in num_cols:
        if c in bin_cols:
            continue
        try:
            nunique = pd.Series(X[c]).dropna().nunique()
        except Exception:
            nunique = None
        if nunique is not None and nunique <= 2:
            bin_cols.append(c)
    # eliminar duplicados preservando orden
    bin_cols = list(dict.fromkeys(bin_cols))
    # numéricos continuos = num_cols - bin_cols
    cont_num: List[str] = [c for c in num_cols if c not in bin_cols]

    # --- Pipelines numéricos (continuos y binarios) ---
    # Continuos: imputación + (opcional) Power + (opcional) StandardScaler + (opcional) VarThresh
    num_steps_cont: List[tuple] = [("imputer", SimpleImputer(strategy=numeric_imputer))]
    if power_transform:
        num_steps_cont.append(("power", PowerTransformer(method="yeo-johnson")))
    if family in scale_for:
        num_steps_cont.append(("scaler", StandardScaler()))
    if drop_low_variance:
        num_steps_cont.append(("varthresh", VarianceThreshold(threshold=low_variance_threshold)))

    # Binarios: SOLO imputación (no Power/Scaler) para no distorsionar indicadores
    bin_steps: List[tuple] = [("imputer", SimpleImputer(strategy=numeric_imputer))]

    # --- Pipelines categóricos ---
    # OneHot denso por defecto; si se reducirá con SVD, pedir sparse para eficiencia.
    want_sparse_ohe = bool(reduce_sparse and (family in ["linear", "logistic", "mlp"]))
    cat_steps: List[tuple] = [("imputer", SimpleImputer(strategy=categorical_imputer, fill_value="Missing"))]
    if rare_min_freq and rare_min_freq > 0:
        cat_steps.append(("rare", RareCategoryGrouper(min_freq=rare_min_freq)))
    # Si LGBM y flag activo => evitar OHE, usamos OrdinalEncoder estable
    if one_hot_for and (family in one_hot_for) and not (family == "lgbm" and use_native_categoricals_for_lgbm):
        cat_steps.append(("onehot", _make_ohe_compat(want_sparse=want_sparse_ohe)))
    else:
        cat_steps.append(("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

    # Pre-pipeline (mapea ordinales y aplica freq-encoding en todo el DF salvo hashed)
    pre = Pipeline(steps=[("ordmap", ord_mapper)])
    if freq_enc is not None:
        pre.steps.append(("freqenc", freq_enc))

    # ColumnTransformer con (num_cont, num_bin, cat, hash?)
    transformers = [
        ("num", Pipeline(num_steps_cont), cont_num),
        ("bin", Pipeline(bin_steps), bin_cols),
        ("cat", Pipeline(cat_steps), cat_cols)
    ]
    if hash_text and len(hashed_cols) > 0:
        transformers.append(("hash", HashingTokenTransformer(cols=hashed_cols, n_features=hash_n_features), hashed_cols))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    pre.steps.append(("ct", ct))

    # Reducción de dimensionalidad para OHE grandes (solo lineales/MLP)
    if reduce_sparse and want_sparse_ohe:
        pre.steps.append(("svd", SVDWrapper(n_components=svd_components, random_state=42)))

    # SelectKBest (se mantiene al final)
    if select_k_best is not None and select_k_best > 0:
        score_func = mutual_info_classif if task == "classification" else mutual_info_regression
        pre.steps.append(("selectk", SelectKBest(score_func=score_func, k=select_k_best)))

    return pre
