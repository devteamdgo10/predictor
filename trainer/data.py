# trainer/data.py
from __future__ import annotations
from typing import Tuple, List, Optional
import pandas as pd
from .utils import load_csv, guess_id_columns, logger


def prepare_dataframe(csv_path: str,
                      target: str,
                      features: Optional[List[str]] = None,
                      drop_id_like: bool = True,
                      id_patterns: Optional[List[str]] = None,
                      high_na_threshold: float = 0.95,
                      sample_rows: Optional[int] = None
                      ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Carga el CSV y regresa (df_original_saneado, y, X) con mejoras seguras:
      - Limpia espacios en nombres de columnas.
      - Normaliza vacíos ""/espacios a NA (para imputación posterior).
      - No permite que la regla de 'ultra-escasas' elimine el target.
      - (Opcional) remueve columnas tipo ID por patrones.
      - Respeta 'features' si se proveen (valida columnas faltantes).
    """
    # 1) Carga
    df = load_csv(csv_path, sample_rows=sample_rows)

    # 1.1 Nombres: quitar espacios alrededor (muy común en CSVs)
    original_cols = list(df.columns)
    df.columns = df.columns.astype(str).str.strip()
    if original_cols != list(df.columns):
        logger.info("Column names were stripped of leading/trailing spaces.")

    # 1.2 Normalizar vacíos a NA (para que imputación funcione correctamente)
    #    Solo reemplaza strings vacíos/espacios, no toca valores '0' o similares.
    before_na = int(df.isna().sum().sum())
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    after_na = int(df.isna().sum().sum())
    if after_na > before_na:
        logger.info(f"Converted empty strings to NA: +{after_na - before_na} NA cells.")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV columns: {list(df.columns)}")

    # 2) Drop de columnas ultra-escasas (excluye target por seguridad)
    #    Calcula el ratio NA sin considerar la columna objetivo.
    na_frac = df.drop(columns=[target], errors="ignore").isna().mean()
    to_drop_sparse = list(na_frac[na_frac > high_na_threshold].index)
    if to_drop_sparse:
        logger.info(f"Dropping ultra-sparse columns (> {high_na_threshold*100:.0f}% NA): {to_drop_sparse}")
        df = df.drop(columns=to_drop_sparse)

    # 3) Separar y / X
    y = df[target]
    X = df.drop(columns=[target])

    # 4) Subconjunto de features (si se proveen)
    if features is not None:
        missing = set(features) - set(X.columns)
        if missing:
            raise ValueError(f"Requested feature columns not found: {missing}")
        X = X[features]

    # 5) Columnas tipo ID (heurística por patrones) — opcional
    if drop_id_like and id_patterns:
        id_cols = guess_id_columns(X, id_patterns)
        if id_cols:
            logger.info(f"Dropping ID-like columns: {id_cols}")
            X = X.drop(columns=id_cols)

    # 6) Log final de dimensiones
    try:
        logger.info(f"Prepared dataframe -> X shape: {X.shape}, y length: {len(y)}")
    except Exception:
        pass

    return df, y, X
