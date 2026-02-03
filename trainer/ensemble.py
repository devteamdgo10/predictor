# trainer/ensemble.py
from __future__ import annotations
from typing import List, Optional
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge

# =========================
# Utilidades internas
# =========================
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def _softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def _predict_proba_safely(est, X) -> np.ndarray:
    """
    Devuelve probabilidades para clasificadores.
    Intento en orden: predict_proba -> decision_function->sigmoid/softmax.
    Lanza si no es posible.
    """
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)
        # Garantizar 2D
        proba = np.asarray(proba)
        if proba.ndim == 1:
            proba = np.c_[1 - proba, proba]
        return proba
    if hasattr(est, "decision_function"):
        df = est.decision_function(X)
        df = np.asarray(df)
        if df.ndim == 1:  # binario
            p1 = _sigmoid(df)
            return np.c_[1 - p1, p1]
        # multiclase
        return _softmax(df, axis=1)
    raise ValueError(f"Estimator {type(est).__name__} no expone predict_proba/decision_function.")

# =========================
# BlendEnsemble (fallback por defecto)
# =========================
class BlendEnsemble:
    """
    Promedio ponderado simple de predicciones de varios estimadores.
    - Para regresión: promedio de predict().
    - Para clasificación: promedio de predict_proba() y argmax para predict().
    """
    def __init__(self, estimators: List, weights: Optional[List[float]] = None, task: str = "classification"):
        self.estimators = list(estimators)
        if weights is None or len(weights) != len(self.estimators):
            weights = [1.0] * len(self.estimators)
        w = np.asarray(weights, dtype=float)
        s = np.sum(w)
        self.weights = (w / s) if s > 0 else np.asarray([1.0 / max(1, len(w))] * len(w))
        self.task = task
        self.classes_ = None  # para clasificación (si procede)

    def fit(self, X, y):
        """
        Intenta ajustar los estimadores si soportan fit.
        Si ya están ajustados, esto no debería romper.
        """
        for est in self.estimators:
            if hasattr(est, "fit") and not hasattr(est, "classes_") and self.task == "classification":
                try:
                    est.fit(X, y)
                except Exception:
                    pass
            elif hasattr(est, "fit") and self.task != "classification":
                try:
                    est.fit(X, y)
                except Exception:
                    pass
        if self.task == "classification":
            # Intentar determinar clases a partir del primero que tenga 'classes_'
            for est in self.estimators:
                if hasattr(est, "classes_"):
                    self.classes_ = getattr(est, "classes_", None)
                    break
        return self

    def _avg_proba(self, X) -> np.ndarray:
        probas = []
        for est in self.estimators:
            proba = _predict_proba_safely(est, X)
            probas.append(proba)
        P = np.zeros_like(probas[0], dtype=float)
        for w, p in zip(self.weights, probas):
            P += w * p
        return P

    def predict_proba(self, X):
        if self.task != "classification":
            raise AttributeError("predict_proba sólo aplica a clasificación.")
        return self._avg_proba(X)

    def predict(self, X):
        if self.task == "classification":
            P = self._avg_proba(X)
            idx = np.argmax(P, axis=1)
            if self.classes_ is not None and len(self.classes_) == P.shape[1]:
                return np.asarray(self.classes_)[idx]
            # Fallback a índices (0..K-1)
            return idx
        # Regresión
        preds = []
        for est in self.estimators:
            preds.append(est.predict(X))
        preds = np.column_stack(preds)
        return np.average(preds, axis=1, weights=self.weights)

# =========================
# StackingEnsemble (opcional)
# =========================
class StackingEnsemble:
    """
    Stacking con meta-modelo (LogisticRegression para clasificación, Ridge para regresión).
    - Genera OOF de los modelos base con CV.
    - Para clasificación: usa probabilidades (o decision_function->sigmoid/softmax).
    - Para regresión: usa predicción continua.
    - Reajusta cada base sobre todo el dataset al final para predicción en test.
    """
    def __init__(
        self,
        estimators: List,
        task: str = "classification",
        meta_model=None,
        cv: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.base_estimators = list(estimators)
        self.task = task
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_jobs = n_jobs

        if meta_model is None:
            if task == "classification":
                self.meta_model = LogisticRegression(max_iter=2000, solver="lbfgs")
            else:
                self.meta_model = Ridge(alpha=1.0, random_state=random_state)
        else:
            self.meta_model = meta_model

        self.fitted_estimators_: List = []
        self.classes_ = None  # para clasificación si aplica

    def _cv_splitter(self, y):
        if self.task == "classification":
            return StratifiedKFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state)
        return KFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state)

    def _estimator_output_dim(self, est, X, y):
        """
        Determina cuántas columnas aporta cada base:
          - Clasificación: n_clases (a partir de predict_proba/decision_function).
          - Regresión: 1
        """
        if self.task == "regression":
            return 1
        # clasificación
        try:
            # Clonar y entrenar en un subset pequeño para checar dimensión
            est_tmp = clone(est)
            n_small = min(64, len(y))
            est_tmp.fit(X[:n_small], y[:n_small])
            proba = _predict_proba_safely(est_tmp, X[:n_small])
            return proba.shape[1]
        except Exception:
            # fallback conservador: binario
            return 2

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Preparar meta-features OOF
        splitter = self._cv_splitter(y)

        # Calcular total de columnas de meta-features
        dims = [self._estimator_output_dim(est, X, y) for est in self.base_estimators]
        total_cols = int(np.sum(dims))
        X_meta = np.zeros((X.shape[0], total_cols), dtype=float)

        col_start = 0
        for est, d in zip(self.base_estimators, dims):
            col_end = col_start + d
            # OOF para este estimador
            oof = np.zeros((X.shape[0], d), dtype=float)

            for tr_idx, va_idx in splitter.split(X, y):
                est_fold = clone(est)
                est_fold.fit(X[tr_idx], y[tr_idx])
                if self.task == "classification":
                    proba = _predict_proba_safely(est_fold, X[va_idx])
                    oof[va_idx] = proba
                else:
                    pred = est_fold.predict(X[va_idx]).reshape(-1, 1)
                    oof[va_idx, 0] = pred[:, 0]

            X_meta[:, col_start:col_end] = oof
            col_start = col_end

        # Ajustar meta-modelo
        self.meta_model.fit(X_meta, y)

        # Ajustar bases sobre todo el dataset (para inferencia futura)
        self.fitted_estimators_ = []
        for est in self.base_estimators:
            est_full = clone(est)
            est_full.fit(X, y)
            self.fitted_estimators_.append(est_full)

        if self.task == "classification":
            # Tratar de fijar classes_ desde meta_model si existe
            if hasattr(self.meta_model, "classes_"):
                self.classes_ = getattr(self.meta_model, "classes_", None)

        return self

    def _stack_features(self, X) -> np.ndarray:
        """
        Construye las features para meta_model a partir de los estimadores entrenados sobre todo el dataset.
        """
        X = np.asarray(X)
        feats = []
        for est in self.fitted_estimators_:
            if self.task == "classification":
                proba = _predict_proba_safely(est, X)
                feats.append(proba)
            else:
                pred = est.predict(X).reshape(-1, 1)
                feats.append(pred)
        return np.concatenate(feats, axis=1)

    def predict_proba(self, X):
        if self.task != "classification":
            raise AttributeError("predict_proba sólo aplica a clasificación.")
        Z = self._stack_features(X)
        if hasattr(self.meta_model, "predict_proba"):
            proba = self.meta_model.predict_proba(Z)
            proba = np.asarray(proba)
            if proba.ndim == 1:
                proba = np.c_[1 - proba, proba]
            return proba
        if hasattr(self.meta_model, "decision_function"):
            df = self.meta_model.decision_function(Z)
            df = np.asarray(df)
            if df.ndim == 1:
                p1 = _sigmoid(df)
                return np.c_[1 - p1, p1]
            return _softmax(df, axis=1)
        # Fallback si el meta-modelo sólo tiene predict(): aproximar con etiquetas duras
        # No hay probabilidades confiables -> producir one-hot de predicción dura
        y_pred = self.meta_model.predict(Z)
        y_pred = np.asarray(y_pred)
        classes = self.classes_ if self.classes_ is not None else np.unique(y_pred)
        # mapear a índices
        proba = np.zeros((len(y_pred), len(classes)), dtype=float)
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        for i, c in enumerate(y_pred):
            proba[i, cls_to_idx.get(c, 0)] = 1.0
        return proba

    def predict(self, X):
        if self.task == "classification":
            P = self.predict_proba(X)
            idx = np.argmax(P, axis=1)
            if self.classes_ is not None and len(self.classes_) == P.shape[1]:
                return np.asarray(self.classes_)[idx]
            return idx
        # Regresión
        Z = self._stack_features(X)
        return self.meta_model.predict(Z)
