import re
from dataclasses import dataclass


@dataclass
class Config:
    data_root: str = "."
    train_dir: str | None = "train"
    val_dir: str | None = "test"
    output_dir: str = "./output"
    checkpoint_dir: str | None = None
    overfit_sample_idx: int | None = None
    device: str | None = "cuda"
    seed: int = 42
    deterministic: bool = False

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


CFG = Config()
DEFAULT_DEVICE = "cuda"
_LAYER_RE = re.compile(r"(\d+)\.pt$")
EPS = 1e-8
