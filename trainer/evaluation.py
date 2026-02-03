# trainer/evaluation.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance

# =========================
# Clasificación
# =========================
def evaluate_classification(y_true, y_pred, y_proba=None, average="macro") -> Dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average=average))
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                p = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                out["roc_auc"] = float(roc_auc_score(y_true, p))
                out["avg_precision"] = float(average_precision_score(y_true, p))
                out["log_loss"] = float(log_loss(y_true, np.c_[1-p, p] if y_proba.ndim==1 else y_proba))
            else:
                out["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                out["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            pass
    return out

# Mantener importaciones separadas como estaban en tu archivo
import numpy as np #as _np_alias  # alias interno para evitar conflictos de reimport
import pandas as pd
from sklearn.metrics import f1_score as _f1_score_alias

def best_threshold_by_f1(y_true, y_proba, num_thresholds: int = 101):
    """
    Devuelve (best_threshold, best_f1) para binario.
    - Acepta y_true con etiquetas tipo string ('Y'/'N', 'Yes'/'No', etc.) o numéricas/bool.
    - y_proba puede ser:
        * 1D: probabilidad de la clase positiva
        * 2D con 2 columnas: probas por clase; se detecta automáticamente la columna de la clase positiva
    """
    y_true = np.asarray(y_true)

    # --- 1) Determinar etiqueta positiva de forma genérica ---
    uniq = pd.Series(y_true).dropna().unique()
    if len(uniq) != 2:
        raise ValueError("best_threshold_by_f1 requiere un problema binario (2 clases).")

    # Heurística de etiqueta positiva
    preferred = [1, "1", True, "Y", "y", "Yes", "YES", "Si", "SI", "S", "True", "TRUE"]
    pos_label = None
    for cand in preferred:
        if cand in uniq:
            pos_label = cand
            break
    if pos_label is None:
        # Si no hay preferida, usar la minoritaria
        vc = pd.Series(y_true).value_counts()
        pos_label = vc.idxmin()

    y_bin = (pd.Series(y_true) == pos_label).to_numpy().astype(int)

    # --- 2) Obtener vector de probas para la clase positiva ---
    p = np.asarray(y_proba)
    if p.ndim == 2:
        if p.shape[1] != 2:
            raise ValueError("Para multiclass (>2) no hay umbral único; se esperaba 2 columnas.")
        # Elegir la columna que mejor correlacione con y_bin (más robusto que asumir orden)
        try:
            corr0 = np.corrcoef(p[:, 0], y_bin)[0, 1]
            corr1 = np.corrcoef(p[:, 1], y_bin)[0, 1]
            pos_col = 1 if (np.nan_to_num(corr1) >= np.nan_to_num(corr0)) else 0
        except Exception:
            # Fallback seguro
            pos_col = 1
        p = p[:, pos_col]
    elif p.ndim != 1:
        raise ValueError("y_proba debe ser un vector 1D o matriz 2D de 2 columnas.")

    # --- 3) Barrer umbrales y elegir el mejor F1 ---
    thrs = np.linspace(0.0, 1.0, num_thresholds)
    f1s = []
    for t in thrs:
        y_pred_bin = (p >= t).astype(int)
        f1s.append(_f1_score_alias(y_bin, y_pred_bin))
    best_idx = int(np.argmax(f1s))
    return float(thrs[best_idx]), float(f1s[best_idx])


def plot_roc_curve(y_true, y_proba, path: str):
    if y_proba is None:
        return
    plt.figure()
    p = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    fpr, tpr, _ = roc_curve(y_true, p)
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
    plt.savefig(path, bbox_inches="tight"); plt.close()

def plot_pr_curve(y_true, y_proba, path: str):
    if y_proba is None:
        return
    plt.figure()
    p = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    pr, rc, _ = precision_recall_curve(y_true, p)
    plt.plot(rc, pr, label="PR")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend()
    plt.savefig(path, bbox_inches="tight"); plt.close()

def plot_confusion(y_true, y_pred, path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(); disp.plot(); plt.title("Confusion Matrix")
    plt.savefig(path, bbox_inches="tight"); plt.close()


# =========================
# Regresión
# =========================
def evaluate_regression(y_true, y_pred) -> Dict[str, Any]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# =========================
# Importancia por permutación (simple)
# =========================
def compute_permutation_importance(estimator, X, y, path: Optional[str]=None) -> Dict[str, float]:
    """
    Calcula importancias por permutación (ligero). Si 'path' se provee, guarda un gráfico Top-20.
    Devuelve dict {feat_i: importance}.
    Nota: si el estimator es un Pipeline, las importancias son sobre las features transformadas.
    """
    try:
        r = permutation_importance(estimator, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = {f"feat_{i}": float(v) for i, v in enumerate(r.importances_mean)}
        if path:
            idx = np.argsort(r.importances_mean)[::-1][:20]
            plt.figure()
            plt.bar(range(len(idx)), r.importances_mean[idx])
            plt.title("Permutation Importance (Top 20)")
            plt.savefig(path, bbox_inches="tight"); plt.close()
        return importances
    except Exception:
        return {}


# =========================
# NUEVO: Curva de aprendizaje (opcional)
# =========================
def learning_curve_plot(estimator, X, y, path: str, cv, scoring: Optional[str] = None,
                        train_sizes: Optional[np.ndarray] = None, n_jobs: int = -1) -> Optional[Dict[str, Any]]:
    """
    Genera curva de aprendizaje (train/val) y la guarda en 'path'.
    Retorna dict con promedios y std si todo sale bien; sino None.
    """
    try:
        from sklearn.model_selection import learning_curve
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)
        sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes
        )
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std     = val_scores.mean(axis=1), val_scores.std(axis=1)

        plt.figure()
        plt.fill_between(sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, label="Train ±1σ")
        plt.fill_between(sizes, val_mean-val_std,   val_mean+val_std,   alpha=0.15, label="CV ±1σ")
        plt.plot(sizes, train_mean, marker="o", label="Train")
        plt.plot(sizes, val_mean,   marker="o", label="CV")
        plt.xlabel("Training examples"); plt.ylabel(scoring or "Score")
        plt.title("Learning Curve")
        plt.legend()
        plt.savefig(path, bbox_inches="tight"); plt.close()

        return {
            "train_sizes": sizes.tolist(),
            "train_mean": train_mean.tolist(),
            "train_std": train_std.tolist(),
            "val_mean": val_mean.tolist(),
            "val_std": val_std.tolist()
        }
    except Exception:
        return None


# =========================
# NUEVO: IC para CV (95% por defecto)
# =========================
def cv_confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Devuelve (mean, low, high) para un vector de scores de CV.
    Usa t-student si SciPy está disponible; si no, aproximación normal.
    """
    scores = np.asarray(scores, dtype=float)
    m = float(scores.mean())
    s = float(scores.std(ddof=1)) if scores.size > 1 else 0.0
    n = scores.size
    if n < 2 or s == 0.0:
        return m, m, m
    try:
        from scipy.stats import t
        alpha = 1.0 - confidence
        tval = float(t.ppf(1 - alpha/2, df=n-1))
        half = tval * s / np.sqrt(n)
    except Exception:
        # Normal approx
        from math import sqrt
        z = 1.96 if abs(confidence - 0.95) < 1e-6 else 1.96  # fijo 95% por simplicidad si no hay scipy
        half = z * s / sqrt(n)
    return m, m - half, m + half


# =========================
# NUEVO: Sugerencia de poda por permutación (ligero)
# =========================
def prune_by_permutation(estimator, X, y, n_repeats: int = 5, random_state: int = 42, n_jobs: int = -1,
                         drop_below: float = 0.0, max_drop: Optional[int] = None,
                         path: Optional[str] = None) -> Dict[str, Any]:
    """
    Calcula importancias por permutación sobre (X,y) para el estimator (puede ser Pipeline).
    SUGIERE descartar features con importancia media <= drop_below (e.g. 0.0).
    No modifica el estimator; sólo devuelve la sugerencia.

    Retorna:
        {
          "indices_to_drop": [i, ...],
          "feature_names_to_drop": ["...", ...] (si disponibles),
          "importances_mean": [...],
          "importances_std":  [...],
        }
    """
    out = {"indices_to_drop": [], "feature_names_to_drop": [], "importances_mean": [], "importances_std": []}
    try:
        r = permutation_importance(estimator, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        imp = r.importances_mean
        std = r.importances_std
        out["importances_mean"] = imp.tolist()
        out["importances_std"] = std.tolist()

        # candidatos por umbral
        cand_idx = np.where(imp <= float(drop_below))[0]
        # si se limita por cantidad, tomar los menos importantes
        if max_drop is not None and len(cand_idx) > max_drop:
            order = np.argsort(imp)  # ascendente
            cand_idx = order[:max_drop]

        out["indices_to_drop"] = cand_idx.tolist()

        # intentar mapear a nombres si es Pipeline con step 'pre'
        feat_names = None
        try:
            from sklearn.pipeline import Pipeline as _SkPipeline
            if isinstance(estimator, _SkPipeline):
                pre = estimator.named_steps.get("pre", None)
                if pre is not None and hasattr(pre, "get_feature_names_out"):
                    feat_names = list(pre.get_feature_names_out())
        except Exception:
            feat_names = None

        if feat_names is not None:
            names = [feat_names[i] for i in cand_idx if i < len(feat_names)]
            out["feature_names_to_drop"] = names

        # gráfico opcional (Top-20)
        if path:
            idx_desc = np.argsort(imp)[::-1][:20]
            plt.figure()
            plt.bar(range(len(idx_desc)), imp[idx_desc], yerr=std[idx_desc])
            plt.title("Permutation Importance (Top 20)")
            plt.savefig(path, bbox_inches="tight"); plt.close()

        return out
    except Exception:
        return out
