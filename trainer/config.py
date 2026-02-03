# trainer/config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# =========================
# Secciones de configuración
# =========================
@dataclass
class DataConfig:
    csv_path: str = ""                         # Ruta al CSV de entrenamiento
    target: Optional[str] = None               # Columna objetivo
    features: Optional[List[str]] = None       # Subconjunto de features (None = todas menos target)

    # Heurísticas genéricas
    drop_id_like: bool = True                  # Quitar columnas con patrón de ID
    id_patterns: List[str] = field(default_factory=lambda: [
        "id", "folio", "uuid", "guid", "numero", "account", "cuenta",
        "no_", "no.", "número", "id_", "_id"
    ])
    high_na_threshold: float = 0.95            # Drop de columnas con NA ratio > umbral
    sample_rows: Optional[int] = None          # Tomar muestra de filas (None = todo)


@dataclass
class PreprocessConfig:
    # Estrategias “auto” por defecto
    handle_nulls: str = "auto"                 # (informativo; la lógica está en data/features)
    power_transform: bool = True               # Activar si hay skew alto
    scale_for: Optional[List[str]] = field(default_factory=lambda: ["linear", "logistic", "mlp"])
    one_hot_for: Optional[List[str]] = None    # None = dejar a la heurística del preprocesador

    # Alta cardinalidad
    freq_encode_high_card: bool = True
    high_card_threshold: int = 50

    # Imputación
    numeric_imputer: str = "median"
    categorical_imputer: str = "most_frequent"

    # Variancia/correlación (opcionales; conservadores por compatibilidad)
    drop_low_variance: bool = False
    low_variance_threshold: float = 0.0
    select_k_best: Optional[int] = None        # None/0/“auto” → heurística decidirá k

    correlation_filter: bool = True            # Si tu pipeline lo usa
    correlation_threshold: float = 0.98

    # Mapeos ordinales si aplica (por nombre de columna)
    ordinal_mappings: Optional[Dict[str, Dict[str, int]]] = None


@dataclass
class CVConfig:
    folds: int = 5
    shuffle: bool = True
    random_state: int = 42
    n_iter_search: int = 30                    # Presupuesto base de RandomizedSearch
    n_jobs: int = -1                           # Usa todos los cores disponibles


@dataclass
class TrainConfig:
    # Tarea/modo
    task: Optional[str] = None                 # "classification"/"regression" o None (auto)
    mode: str = "auto"                         # "auto" o "custom"
    model_names: Optional[List[str]] = None    # En modo custom

    # Métricas y balanceo
    class_weight: Optional[str] = None         # None → heurística; "balanced" para forzar
    calibrate: bool = True                     # Deseable para probas calibradas (si el flujo lo usa)
    primary_metric: Optional[str] = None       # None → heurística (roc_auc / f1_macro / RMSE)
    secondary_metrics: List[str] = field(default_factory=list)

    # Ensamble y aceleración
    ensemble_top_n: int = 1                    # Compat: el pipeline actual usa 1
    use_gpu_if_available: bool = True

    # Salidas
    output_dir: str = "outputs"
    save_pipeline_as: str = "pipeline.joblib"
    save_report_as: str = "report.json"
    save_readme: bool = True

    # Nuevas banderas “auto” (opt-in desde auto_decider / train)
    search_strategy: Optional[str] = None      # "random"/"halving" (informativo)
    early_stopping: Optional[bool] = True
    enable_cache: bool = True
    auto_plan: bool = True                     # Habilita sugerencias automáticas (auto_decider)
    save_preprocessed_dataset: bool = True     # Exporta X_preprocessed + y a CSV


# =========================
# Configuración raíz
# =========================
@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Utilitarios
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SystemConfig":
        """Crea SystemConfig desde un dict (p.ej., cargado de JSON). Campos faltantes usan defaults."""
        data = DataConfig(**d.get("data", {}))
        preprocess = PreprocessConfig(**d.get("preprocess", {}))
        cv = CVConfig(**d.get("cv", {}))
        train = TrainConfig(**d.get("train", {}))
        return SystemConfig(data=data, preprocess=preprocess, cv=cv, train=train)
