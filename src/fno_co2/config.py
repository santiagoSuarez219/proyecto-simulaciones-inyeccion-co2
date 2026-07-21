import re
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    data_root: str = "data/processed"
    train_dir: str | None = "train"
    val_dir: str | None = "test"
    output_dir: str = "./output"
    checkpoint_dir: str | None = None
    overfit_sample_idx: int | None = None
    device: str | None = "cuda"
    seed: int = 42
    deterministic: bool = False
    experiment_name: str = "baseline"
    model_variant: str = "fno_baseline"

    batch_size: int = 4
    epochs: int = 100
    lr: float = 8e-4
    lr_scheduler: str | None = "cosine"
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    num_workers: int | None = None
    prefetch_factor: int = 2
    persistent_workers: bool = True
    progress_interval: int = 10
    grad_clip: float = 1.0
    use_amp: bool = False

    time_steps: int = 61
    hidden_dim: int = 128
    spectral_modes: int = 16
    dropout_p: float = 0.1
    use_group_norm: bool = False  # M3, EXPERIMENTAL — ver docstring en models/blocks.py::ResBlock
    unet_depth: int = 3  # Solo afecta a la variante unet_film; baseline lo ignora
    attn_heads: int = 4  # spec-003: número de cabezas de atención axial (solo afecta fno_axial_attn)
    attn_num_blocks: int = 4  # spec-003: cuántos de los 4 bloques llevan atención intercalada

    auto_resume: bool = True
    pause_hour: int = 7
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    sf_weight: float = 2.5
    vd_weight: float = 1.0
    grad_weight: float = 0.8
    seg_t0_weight: float = 3.0
    seg_t1_20_weight: float = 2.0
    seg_t21_60_weight: float = 1.0

    save_epoch_pngs: bool = False
    epoch_png_examples: int = 1
    uncertainty_passes: int = 30
    # Cada cuantas epocas se recalibra y se computa la incertidumbre MC-Dropout completa
    # (cara: uncertainty_passes forwards sobre val). El val_loss/R²/RMSE y la seleccion de
    # best.pt usan SIEMPRE el forward determinista por epoca; la incertidumbre es un
    # diagnostico periodico. <=0 => solo en la epoca final. La epoca final siempre se computa.
    uncertainty_eval_interval: int = 10


def load_config_from_yaml(path: str | Path) -> Config:
    """Carga un `Config` desde un YAML autocontenido (spec-001 Fase 2). El archivo debe
    declarar solo campos existentes en el dataclass `Config` — cualquier clave desconocida
    o campo faltante hace fallar la carga explícito en vez de silenciarlo."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    valid_keys = {f.name for f in fields(Config)}
    unknown_keys = set(data) - valid_keys
    if unknown_keys:
        raise ValueError(
            f"{path}: claves desconocidas para Config: {sorted(unknown_keys)}"
        )

    return Config(**data)


CFG = Config()
DEFAULT_DEVICE = "cuda"
_LAYER_RE = re.compile(r"(\d+)\.pt$")
EPS = 1e-8
