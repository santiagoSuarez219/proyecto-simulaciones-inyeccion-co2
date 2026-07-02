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

    batch_size: int = 4
    epochs: int = 100
    lr: float = 8e-4
    weight_decay: float = 1e-4
    num_workers: int | None = None
    prefetch_factor: int = 2
    persistent_workers: bool = True
    progress_interval: int = 10
    grad_clip: float = 1.0

    time_steps: int = 61
    hidden_dim: int = 128
    spectral_modes: int = 16

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


CFG = Config()
DEFAULT_DEVICE = "cuda"
_LAYER_RE = re.compile(r"(\d+)\.pt$")
EPS = 1e-8
