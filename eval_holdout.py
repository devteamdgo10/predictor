# eval_holdout.py
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import joblib
import numpy as np
import pandas as pd

# sklearn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import NotFittedError

# plotting
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ---------- Utilidades ----------

def _load_pipeline(run_dir: Path) -> Any:
    """Carga el artefacto de modelo entrenado (por defecto pipeline.joblib)."""
    cand = run_dir / "pipeline.joblib"
    if cand.exists():
        return joblib.load(cand)
    # fallback: primer .joblib
    jobs = list(run_dir.glob("*.joblib"))
    if not jobs:
        raise FileNotFoundError(f"No .joblib found in {run_dir}")
    return joblib.load(jobs[0])


def _load_report(run_dir: Path) -> Optional[Dict[str, Any]]:
    rp = run_dir / "report.json"
    if not rp.exists():
        return None
    try:
        with rp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_report_task(run_dir: Path) -> Optional[str]:
    rep = _load_report(run_dir)
    if not rep:
        return None
    return rep.get("task")


def _read_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _align_columns_like_preprocessor(model: Any, X: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """
    Intenta reordenar/agregar columnas (con NaN) para que coincidan con lo visto en entrenamiento.
    Busca feature_names_in_ en el preprocesador.
    """
    expected_cols: Optional[List[str]] = None

    try:
        # Pipeline(pre, model)
        pre = model.named_steps.get("pre", None)
        if pre is not None and hasattr(pre, "feature_names_in_"):
            expected_cols = list(pre.feature_names_in_)
    except Exception:
        pass

    if expected_cols is not None:
        # reindex en el mismo orden; columnas faltantes como NaN; extras se ignoran
        X = X.reindex(columns=expected_cols)
    return X, expected_cols


def _to_proba_vector(y_proba: np.ndarray) -> np.ndarray:
    """Para binario: regresa prob. de la clase positiva. Si (n,2) -> col 1; si (n,) la misma."""
    if y_proba is None:
        return None
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return y_proba[:, 1]
    return y_proba


def _compute_metrics_classification(y_true: Optional[pd.Series],
                                    y_pred: np.ndarray,
                                    y_proba: Optional[np.ndarray],
                                    classes_: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Calcula métricas de clasificación. Si y_true es None, se devuelven métricas vacías.
    Maneja binario y multiclase (macro).
    """
    if y_true is None:
        return {}

    y_true = pd.Series(y_true).values
    metrics: Dict[str, Any] = {}

    # accuracy / f1_macro siempre disponibles
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        pass
    try:
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        pass

    # ROC-AUC y Avg Precision requieren scores/probas
    if y_proba is not None:
        try:
            # binario vs multiclase
            if classes_ is None or len(np.unique(y_true)) <= 2:
                y_score = _to_proba_vector(y_proba)
                if y_score is not None and y_score.ndim == 1:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
                    metrics["avg_precision"] = float(average_precision_score(y_true, y_score))
            else:
                # multiclase: macro-ovr
                classes_sorted = np.unique(y_true)
                Y_true_bin = label_binarize(y_true, classes=classes_sorted)
                if Y_true_bin.shape[1] == y_proba.shape[1]:
                    metrics["roc_auc"] = float(roc_auc_score(Y_true_bin, y_proba, average="macro", multi_class="ovr"))
                    # AP macro por clase
                    ap_list = []
                    for j in range(Y_true_bin.shape[1]):
                        ap_list.append(average_precision_score(Y_true_bin[:, j], y_proba[:, j]))
                    metrics["avg_precision"] = float(np.mean(ap_list))
        except Exception:
            pass

    # log_loss si hay probas
    if y_proba is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            pass

    return metrics


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, labels: Optional[List[Any]] = None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title("Holdout - Matriz de confusión")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_roc_binary(y_true: np.ndarray, y_score: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Holdout - Curva ROC (binario)")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pr_binary(y_true: np.ndarray, y_score: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Holdout - Curva Precisión-Recall (binario)")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _silence_feature_name_warning():
    warnings.filterwarnings(
        "ignore",
        message=r"X does not have valid feature names.*fitted with feature names.*"
    )


# ----- NUEVO: soporte labels_info para remapear y_true -----

def _extract_labels_info(run_dir: Path) -> Optional[Dict[str, Any]]:
    rep = _load_report(run_dir)
    if not rep:
        return None
    return rep.get("labels_info")


def _encode_y_true_with_labels_info(y_true: pd.Series,
                                    labels_info: Optional[Dict[str, Any]]) -> pd.Series:
    """
    Si report.json contiene:
      - labels_info: { "mapping": {"Y":1,"N":0} } -> aplica mapping
      - labels_info: { "classes": ["N","Y"] }     -> mapea valor -> índice
    Si no hay info válida, devuelve y_true sin cambios.
    """
    if labels_info is None:
        return y_true

    # Mapping explícito
    mapping = labels_info.get("mapping")
    if isinstance(mapping, dict) and len(mapping) > 0:
        def _map_val(v):
            key = v
            # probar variantes string
            if key not in mapping:
                key = str(v)
            return mapping.get(key, v)
        try:
            y_mapped = y_true.map(_map_val)
            return y_mapped
        except Exception:
            pass

    # Lista de clases ordenadas
    classes = labels_info.get("classes")
    if isinstance(classes, list) and len(classes) > 0:
        idx_map = {cls: i for i, cls in enumerate(classes)}
        def _map_idx(v):
            return idx_map.get(v, idx_map.get(str(v), v))
        try:
            y_mapped = y_true.map(_map_idx)
            return y_mapped
        except Exception:
            pass

    return y_true


# ----- NUEVO: optimización de umbral -----

def _best_threshold_f1(y_true_bin: np.ndarray, y_score: np.ndarray, n: int = 101) -> float:
    thrs = np.linspace(0.0, 1.0, n)
    best_t, best_f1 = 0.5, -1.0
    for t in thrs:
        yp = (y_score >= t).astype(int)
        try:
            f1 = f1_score(y_true_bin, yp)
        except Exception:
            f1 = -1.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def _best_threshold_roc_youden(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true_bin, y_score)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    t = thr[idx]
    # roc_curve puede devolver inf como primer threshold; ajustamos a 1.0 si fuera el caso
    if not np.isfinite(t):
        return 1.0
    # asegurar [0,1]
    return float(np.clip(t, 0.0, 1.0))


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained run on holdout CSV.")
    parser.add_argument("--run-dir", type=str, required=True, help="Ruta del directorio del run (contiene pipeline.joblib).")
    parser.add_argument("--csv", type=str, required=True, help="CSV holdout para evaluar.")
    parser.add_argument("--target", type=str, required=False, help="Nombre de la columna objetivo en el CSV (si existe).")
    parser.add_argument("--plots", action="store_true", help="Genera imágenes: matriz de confusión, ROC y PR (si binario).")
    parser.add_argument("--save-preds", action="store_true", help="Guarda holdout_predictions.csv con y_true/y_pred/probas.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral para convertir probas a clase positiva (binario).")
    parser.add_argument("--optimize", type=str, choices=["none", "f1", "roc"], default="none",
                        help="Optimiza el umbral en holdout (binario): 'f1' o 'roc' (Youden). Por defecto 'none'.")
    parser.add_argument("--quiet-warnings", action="store_true", help="Silencia warnings de nombres de columnas.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = Path(args.csv)
    out_dir = run_dir  # guardamos salidas junto al run

    if args.quiet_warnings:
        _silence_feature_name_warning()

    # 1) Carga modelo y (opcional) task
    model = _load_pipeline(run_dir)
    task_from_report = _load_report_task(run_dir)
    labels_info = _extract_labels_info(run_dir)

    # 2) Carga CSV y separa X / y si target existe
    df_in = _read_csv(csv_path)
    target = args.target
    y_true = None
    if target and target in df_in.columns:
        y_true = df_in[target].copy()
        # NUEVO: remapear y_true si hay labels_info
        try:
            y_true = _encode_y_true_with_labels_info(y_true, labels_info)
        except Exception:
            pass
        X = df_in.drop(columns=[target])
    else:
        X = df_in.copy()

    # 3) Alineación de columnas con el preprocesador de entrenamiento
    X_aligned, expected_cols = _align_columns_like_preprocessor(model, X)

    # 4) Predicciones
    try:
        y_pred_model = model.predict(X_aligned)
    except NotFittedError:
        raise RuntimeError("El pipeline no está entrenado (NotFittedError). Revisa el artefacto en run_dir.")
    except Exception as e:
        # intento de reindex básico en caso de que expected_cols sea None y el estimador requiera nombres
        raise

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_aligned)
        except Exception:
            y_proba = None

    classes_: Optional[np.ndarray] = None
    try:
        classes_ = getattr(model, "classes_", None)
    except Exception:
        classes_ = None

    # 5) Elección de predicciones para evaluar (threshold u optimize si aplica)
    used_threshold: Optional[float] = None
    y_pred_eval = y_pred_model  # por defecto, comportamiento actual

    is_classification = (task_from_report == "classification") or (y_true is not None and pd.Series(y_true).nunique() <= 20)

    if is_classification and y_true is not None and y_proba is not None and pd.Series(y_true).nunique() == 2:
        y_score = _to_proba_vector(y_proba)
        if y_score is not None and y_score.ndim == 1:
            if args.optimize != "none":
                # optimización de umbral en holdout
                y_true_bin = pd.Series(y_true).astype(int).to_numpy()
                if args.optimize == "f1":
                    used_threshold = _best_threshold_f1(y_true_bin, y_score)
                else:  # "roc"
                    used_threshold = _best_threshold_roc_youden(y_true_bin, y_score)
                y_pred_eval = (y_score >= used_threshold).astype(int)
            elif args.threshold is not None:
                used_threshold = float(np.clip(args.threshold, 0.0, 1.0))
                y_pred_eval = (y_score >= used_threshold).astype(int)

    # 6) Métricas
    metrics: Dict[str, Any] = {}
    if is_classification:
        metrics = _compute_metrics_classification(y_true, y_pred_eval, y_proba, classes_)
        if used_threshold is not None:
            metrics["used_threshold"] = float(used_threshold)
    else:
        # regresión (si aplica)
        if y_true is not None:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            try:
                metrics["r2"] = float(r2_score(y_true, y_pred_eval))
            except Exception:
                pass
            try:
                metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred_eval)))
            except Exception:
                pass
            try:
                metrics["mae"] = float(mean_absolute_error(y_true, y_pred_eval))
            except Exception:
                pass

    # 7) Guardar predicciones si se pide
    if args.save_preds:
        out_pred = out_dir / "holdout_predictions.csv"
        out_df = pd.DataFrame()
        if y_true is not None:
            out_df["y_true"] = pd.Series(y_true).values
        out_df["y_pred"] = y_pred_eval
        # guardar proba(s) si existen
        if y_proba is not None:
            if y_proba.ndim == 1:
                out_df["y_score"] = y_proba
            elif y_proba.ndim == 2:
                for j in range(y_proba.shape[1]):
                    out_df[f"proba_class_{j}"] = y_proba[:, j]
        if used_threshold is not None:
            out_df.attrs["used_threshold"] = used_threshold  # metadato ligero (no se serializa en CSV)
        out_df.to_csv(out_pred, index=False)

    # 8) Plots si clasificación binaria y se solicita y existe y_true
    if args.plots and is_classification and y_true is not None:
        labels_sorted = sorted(pd.Series(y_true).unique().tolist())
        # CM (usa y_pred_eval para que respete threshold/optimize si se pidió)
        try:
            _plot_confusion(np.array(y_true), np.array(y_pred_eval), out_dir / "holdout_confusion_matrix.png", labels=labels_sorted)
        except Exception:
            pass
        # ROC / PR solo binario con y_score
        try:
            if pd.Series(y_true).nunique() == 2 and y_proba is not None:
                y_score_vec = _to_proba_vector(y_proba)
                if y_score_vec is not None and y_score_vec.ndim == 1:
                    _plot_roc_binary(np.array(y_true), y_score_vec, out_dir / "holdout_roc_curve.png")
                    _plot_pr_binary(np.array(y_true), y_score_vec, out_dir / "holdout_pr_curve.png")
        except Exception:
            pass

    # 9) Salida JSON
    out = {
        "task": "classification" if is_classification else "regression",
        "metrics": metrics
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
# Ejemplo:
# python eval_holdout.py --run-dir "outputs\\classification_20251103_232719" --csv ".\\datasets\\titanic\\tested.csv" --target Survived --save-preds --quiet-warnings
# Con optimización de umbral F1:
# python eval_holdout.py --run-dir "..." --csv "..." --target ... --optimize f1 --plots --save-preds
