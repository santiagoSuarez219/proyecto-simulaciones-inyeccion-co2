import argparse
import copy
import json
import os
import random
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split


# ==========================================
# CONFIG
# ==========================================
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
_MC_DROPOUT_WARNING_EMITTED = False
_CUDA_BATCH_REPORT_EMITTED = False


# ==========================================
# UTILS
# ==========================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    requested = (requested or DEFAULT_DEVICE).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "gpu":
        requested = "cuda"

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Se pidio usar CUDA/GPU, pero esta instalacion de PyTorch no detecta CUDA. "
                "Revisa que estes ejecutando este archivo con el mismo Python que tiene PyTorch CUDA "
                "o instala una build de PyTorch con soporte CUDA."
            )

        cuda_index = 0 if device.index is None else device.index
        if cuda_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Se pidio {device}, pero solo hay {torch.cuda.device_count()} GPU(s) CUDA disponibles."
            )

        torch.cuda.set_device(cuda_index)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device(f"cuda:{cuda_index}")

    if device.type != "cpu":
        raise ValueError(f"Dispositivo no soportado: {requested}. Usa cuda, cuda:0, gpu, auto o cpu.")
    return device


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = 0 if device.index is None else device.index
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024**3)
        return f"cuda:{idx} | {props.name} | {total_gb:.1f} GB"
    return str(device)


def assert_model_on_device(model: nn.Module, device: torch.device):
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        return

    if model_device.type != device.type:
        raise RuntimeError(f"El modelo quedo en {model_device}, pero se esperaba {device}.")
    if device.type == "cuda" and model_device.index != device.index:
        raise RuntimeError(f"El modelo quedo en {model_device}, pero se esperaba {device}.")


def resolve_num_workers(requested: int | None) -> int:
    if requested is not None:
        return max(0, int(requested))

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return min(8, max(2, cpu_count // 2))


def build_loader(dataset, cfg: Config, device: torch.device, shuffle: bool):
    num_workers = resolve_num_workers(cfg.num_workers)
    kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
    return DataLoader(dataset, **kwargs), num_workers


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path, default=None):
    if not path.exists():
        return [] if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def torch_r2_score(pred, gt):
    ss_res = torch.sum((gt - pred) ** 2)
    ss_tot = torch.sum((gt - torch.mean(gt)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def compute_rmse(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt) + 1e-12)


def spatial_gradient_loss(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


def model_has_dropout(model) -> bool:
    return any(isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) for module in model.modules())


def default_uncertainty_calibration():
    return {
        "sf": {
            "alpha": 0.0,
            "error_q95": 1.0,
        },
        "vd": {
            "alpha": 0.0,
            "error_q95": 1.0,
        },
    }


def predict_with_uncertainty(model, x, d, inj, passes: int):
    global _MC_DROPOUT_WARNING_EMITTED

    if passes <= 1:
        pred = model(x, d, inj)
        return pred, torch.zeros_like(pred)

    was_training = model.training
    model.eval()
    dropout_modules = []
    dropout_prev_training = []
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            dropout_modules.append(module)
            dropout_prev_training.append(module.training)
            module.train(True)

    if not dropout_modules and not _MC_DROPOUT_WARNING_EMITTED:
        print("[UNCERTAINTY WARNING] El modelo no tiene capas Dropout activas; MC Dropout devolvera desviacion estandar cero.")
        _MC_DROPOUT_WARNING_EMITTED = True

    if not dropout_modules:
        with torch.no_grad():
            pred = model(x, d, inj)
        model.train(was_training)
        return pred, torch.zeros_like(pred)

    pred_mean = None
    pred_m2 = None
    with torch.no_grad():
        for pass_idx in range(passes):
            pred = model(x, d, inj)
            if pred_mean is None:
                pred_mean = pred.detach().clone()
                pred_m2 = torch.zeros_like(pred_mean)
                continue

            delta = pred.detach() - pred_mean
            pred_mean = pred_mean + delta / float(pass_idx + 1)
            pred_m2 = pred_m2 + delta * (pred.detach() - pred_mean)

    for module, prev_training in zip(dropout_modules, dropout_prev_training):
        module.train(prev_training)
    model.train(was_training)

    if passes > 1:
        pred_var = pred_m2 / float(passes - 1)
    else:
        pred_var = torch.zeros_like(pred_mean)
    pred_std = torch.sqrt(pred_var.clamp_min(0.0))
    return pred_mean, pred_std


def calibrate_uncertainty(model, val_loader, cfg: Config, device: torch.device):
    if not model_has_dropout(model):
        return default_uncertainty_calibration()

    model.eval()
    sf_std_values = []
    vd_std_values = []
    sf_abs_error_values = []
    vd_abs_error_values = []

    pbar = tqdm(val_loader, desc="calib", leave=False)
    with torch.no_grad():
        for x, d, inj, y in pbar:
            x = x.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)
            inj = inj.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred_mean, pred_std = predict_with_uncertainty(model, x, d, inj, cfg.uncertainty_passes)
            abs_error = (pred_mean - y).abs()

            sf_std_values.append(pred_std[:, :, 0].reshape(-1).cpu())
            vd_std_values.append(pred_std[:, :, 1].reshape(-1).cpu())
            sf_abs_error_values.append(abs_error[:, :, 0].reshape(-1).cpu())
            vd_abs_error_values.append(abs_error[:, :, 1].reshape(-1).cpu())

    sf_std_tensor = torch.cat(sf_std_values) if sf_std_values else torch.zeros(1)
    vd_std_tensor = torch.cat(vd_std_values) if vd_std_values else torch.zeros(1)
    sf_abs_error_tensor = torch.cat(sf_abs_error_values) if sf_abs_error_values else torch.zeros(1)
    vd_abs_error_tensor = torch.cat(vd_abs_error_values) if vd_abs_error_values else torch.zeros(1)

    sf_std_mean = float(sf_std_tensor.mean().item())
    vd_std_mean = float(vd_std_tensor.mean().item())
    sf_abs_error_mean = float(sf_abs_error_tensor.mean().item())
    vd_abs_error_mean = float(vd_abs_error_tensor.mean().item())

    sf_alpha = 0.0 if sf_std_mean <= EPS else float(sf_abs_error_mean / (sf_std_mean + EPS))
    vd_alpha = 0.0 if vd_std_mean <= EPS else float(vd_abs_error_mean / (vd_std_mean + EPS))
    sf_error_q95 = float(torch.quantile(sf_abs_error_tensor, 0.95).item())
    vd_error_q95 = float(torch.quantile(vd_abs_error_tensor, 0.95).item())

    return {
        "sf": {
            "alpha": sf_alpha,
            "error_q95": max(sf_error_q95, EPS),
        },
        "vd": {
            "alpha": vd_alpha,
            "error_q95": max(vd_error_q95, EPS),
        },
    }


def build_uncertainty_map(pred_std, calibration):
    sf_alpha = float(calibration["sf"]["alpha"])
    vd_alpha = float(calibration["vd"]["alpha"])
    sf_error_q95 = max(float(calibration["sf"]["error_q95"]), EPS)
    vd_error_q95 = max(float(calibration["vd"]["error_q95"]), EPS)

    calibrated_std_sf = sf_alpha * pred_std[:, :, 0]
    calibrated_std_vd = vd_alpha * pred_std[:, :, 1]

    uncertainty_sf = (calibrated_std_sf / sf_error_q95).clamp(0.0, 1.0)
    uncertainty_vd = (calibrated_std_vd / vd_error_q95).clamp(0.0, 1.0)
    return torch.stack([uncertainty_sf, uncertainty_vd], dim=2)


def summarize_uncertainty(pred_std, calibration):
    uncertainty_map = build_uncertainty_map(pred_std, calibration)
    confidence_map = 1.0 - uncertainty_map
    sf_unc_flat = uncertainty_map[:, :, 0].reshape(-1)
    vd_unc_flat = uncertainty_map[:, :, 1].reshape(-1)
    return {
        "sf_uncertainty_mean": float(uncertainty_map[:, :, 0].mean().item()),
        "vd_uncertainty_mean": float(uncertainty_map[:, :, 1].mean().item()),
        "sf_confidence_mean": float(confidence_map[:, :, 0].mean().item()),
        "vd_confidence_mean": float(confidence_map[:, :, 1].mean().item()),
        "sf_uncertainty_p95": float(torch.quantile(sf_unc_flat, 0.95).item()),
        "vd_uncertainty_p95": float(torch.quantile(vd_unc_flat, 0.95).item()),
    }


def get_next_pause_datetime(pause_hour: int) -> datetime:
    now = datetime.now()
    pause_dt = now.replace(hour=pause_hour, minute=0, second=0, microsecond=0)
    if now >= pause_dt:
        pause_dt = pause_dt + timedelta(days=1)
    return pause_dt


def resolve_dir(path_value: str | None, data_root: str, fallbacks, label: str) -> Path:
    root = Path(data_root)
    candidates = []

    if path_value:
        p = Path(path_value)
        if not p.is_absolute():
            p = root / p
        candidates.append(p)
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"No existe el directorio de {label}: {p}")

    for name in fallbacks:
        p = root / name
        candidates.append(p)
        if p.exists() and p.is_dir():
            return p

    searched = ", ".join(str(x) for x in candidates) or "(sin candidatos)"
    raise FileNotFoundError(f"No pude resolver el directorio de {label}. Busque en: {searched}")


def build_run_signature(cfg: Config, train_path: Path, val_path: Path):
    return {
        "train_dir": str(train_path.resolve()),
        "val_dir": str(val_path.resolve()),
        "time_steps": int(cfg.time_steps),
        "hidden_dim": int(cfg.hidden_dim),
        "spectral_modes": int(cfg.spectral_modes),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_size),
        "grad_clip": float(cfg.grad_clip),
        "sf_weight": float(cfg.sf_weight),
        "vd_weight": float(cfg.vd_weight),
        "grad_weight": float(cfg.grad_weight),
        "seg_t0_weight": float(cfg.seg_t0_weight),
        "seg_t1_20_weight": float(cfg.seg_t1_20_weight),
        "seg_t21_60_weight": float(cfg.seg_t21_60_weight),
        "overfit_sample_idx": cfg.overfit_sample_idx,
        "model_name": "PhysicalFNOArchitectureRealInjection",
    }


def check_resume_compatibility(saved_signature: dict | None, current_signature: dict):
    if not saved_signature:
        return False, ["checkpoint sin firma de corrida"]

    mismatches = []
    for key, current_value in current_signature.items():
        saved_value = saved_signature.get(key)
        if saved_value != current_value:
            mismatches.append(f"{key}: saved={saved_value} current={current_value}")
    return len(mismatches) == 0, mismatches


def save_training_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer,
    cfg: Config,
    epoch: int,
    best_val_loss: float,
    metrics_row: dict,
    run_signature: dict,
):
    ckpt = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "metrics": metrics_row,
        "run_signature": run_signature,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, ckpt_path)


def try_resume_training(
    ckpt_path: Path,
    model: nn.Module,
    optimizer,
    device: torch.device,
    run_signature: dict,
):
    if not ckpt_path.exists():
        return 1, float("inf"), None, False, [f"checkpoint no existe: {ckpt_path}"], None

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except (RuntimeError, OSError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        return 1, float("inf"), None, False, [f"checkpoint corrupto o ilegible ({ckpt_path.name}): {exc}"], None

    ok, reasons = check_resume_compatibility(ckpt.get("run_signature"), run_signature)
    if not ok:
        return 1, float("inf"), None, False, reasons, None

    try:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except (RuntimeError, KeyError, ValueError) as exc:
        return 1, float("inf"), None, False, [f"estado invalido en {ckpt_path.name}: {exc}"], None

    start_epoch = int(ckpt["epoch"]) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    last_metrics = ckpt.get("metrics")
    return start_epoch, best_val_loss, last_metrics, True, [], ckpt_path


# ==========================================
# DATASET
# ==========================================
def _k(p: Path):
    m = _LAYER_RE.search(p.name)
    return int(m.group(1)) if m else None


def load_pt(p: Path):
    x = torch.load(p, map_location="cpu")
    if isinstance(x, dict):
        for v in x.values():
            if torch.is_tensor(v):
                return v.float()
        raise ValueError(f"Archivo dict sin tensores: {p}")
    if not torch.is_tensor(x):
        raise ValueError(f"Archivo no tensor: {p}")
    return x.float()


def load_injection_series(inj_paths, time_steps: int):
    if not inj_paths:
        return torch.zeros(time_steps, 2, dtype=torch.float32)

    series = []
    for p in sorted(inj_paths)[:2]:
        t = load_pt(p).float().reshape(-1)
        if t.numel() < time_steps:
            pad = torch.zeros(time_steps - t.numel(), dtype=torch.float32)
            t = torch.cat([t, pad], dim=0)
        else:
            t = t[:time_steps]
        series.append(t)

    while len(series) < 2:
        series.append(torch.zeros(time_steps, dtype=torch.float32))

    inj = torch.stack(series[:2], dim=1)
    inj = torch.nan_to_num(inj, nan=0.0, posinf=0.0, neginf=0.0)
    inj = torch.log1p(torch.clamp(inj, min=0.0))
    scale = inj.abs().amax().clamp_min(1e-8)
    inj = inj / scale
    return inj


class DatasetLayers(Dataset):
    def __init__(self, root, max_layer=60):
        self.samples = []
        self.max_layer = int(max_layer)
        self._inj_cache = {}
        root = Path(root)
        if not root.exists():
            return

        for case in root.iterdir():
            if not case.is_dir():
                continue

            static = {
                "AFI": case / "afi_layer_cubes",
                "COH": case / "cohesion_layer_cubes",
                "PERM": case / "permeability_layer_cubes",
                "PORO": case / "porosity_layer_cubes",
            }
            target = case / "layer_cubes"
            inj_dir = case / "injection_name_tensors"

            if not target.exists() or not all(v.exists() for v in static.values()):
                continue

            idx_static = {k: {_k(f): f for f in v.glob("*.pt")} for k, v in static.items()}
            idx_target = {_k(f): f for f in target.glob("*.pt")}
            inj_paths = sorted(inj_dir.glob("*.pt")) if inj_dir.exists() else []

            for k in range(1, 98):
                if k in idx_target and all(k in idx_static[p] for p in static):
                    self.samples.append({
                        "case": case.name,
                        "k": k,
                        "static": {p: idx_static[p][k] for p in static},
                        "target": idx_target[k],
                        "inj_paths": inj_paths,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = torch.stack(
            [load_pt(s["static"][k]).squeeze() for k in ["AFI", "COH", "PERM", "PORO"]],
            dim=0,
        ).float()
        y = load_pt(s["target"]).permute(1, 0, 2, 3)[: self.max_layer + 1].float()
        depth = torch.tensor([(s["k"] - 1) / 96.0], dtype=torch.float32)
        inj_key = tuple(s.get("inj_paths", []))
        inj = self._inj_cache.get(inj_key)
        if inj is None or inj.size(0) != y.shape[0]:
            inj = load_injection_series(s.get("inj_paths", []), y.shape[0])
            self._inj_cache[inj_key] = inj
        return x, depth, inj, y


# ==========================================
# MODEL (FNO + FiLM + real temporal injection)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1, padding_mode="replicate"),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class FiLMSpectralBlock(nn.Module):
    def __init__(self, c, modes, cond_dim=128):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(c, c, modes, modes, dtype=torch.cfloat) * 0.02)
        self.local = nn.Conv2d(c, c, 1)
        self.gamma = nn.Linear(cond_dim, c)
        self.beta = nn.Linear(cond_dim, c)

    def forward(self, x, cond_emb):
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        mh = min(self.modes, x_ft.size(-2))
        mw = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :mh, :mw] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :mh, :mw],
            self.weight[:, :, :mh, :mw],
        )
        spec_x = torch.fft.irfft2(out_ft, s=x.shape[-2:], norm="ortho")

        y = F.gelu(spec_x + self.local(x))
        g = self.gamma(cond_emb).view(-1, y.size(1), 1, 1)
        b = self.beta(cond_emb).view(-1, y.size(1), 1, 1)
        return y * (1.0 + g) + b


class PhysicalFNOArchitecture(nn.Module):
    def __init__(self, time_steps=61, in_c=5, h_dim=128, modes=16, cond_dim=128):
        super().__init__()
        self.time_steps = int(time_steps)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, h_dim, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            ResBlock(h_dim),
        )
        self.t_embed = nn.Embedding(self.time_steps, cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.fno_blocks = nn.ModuleList([FiLMSpectralBlock(h_dim, modes, cond_dim=cond_dim) for _ in range(4)])
        self.decoder = nn.Sequential(
            ResBlock(h_dim),
            nn.Conv2d(h_dim, h_dim // 2, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(h_dim // 2, 2, 1),
        )

    def forward(self, x, d, inj):
        b, _, h, w = x.shape
        depth_map = d.view(b, 1, 1, 1).expand(b, 1, h, w)
        z = self.encoder(torch.cat([x, depth_map], dim=1))

        if inj.ndim == 2:
            inj = inj.unsqueeze(0)
        if inj.size(1) < self.time_steps:
            pad = torch.zeros(b, self.time_steps - inj.size(1), inj.size(2), device=inj.device, dtype=inj.dtype)
            inj = torch.cat([inj, pad], dim=1)
        else:
            inj = inj[:, : self.time_steps]

        t_idx = torch.arange(self.time_steps, device=x.device)
        t_emb = self.t_embed(t_idx).unsqueeze(0).expand(b, self.time_steps, -1)
        depth_seq = d.unsqueeze(1).expand(-1, self.time_steps, -1)
        cond_input = torch.cat([inj, depth_seq], dim=2)
        cond_seq = t_emb + self.cond_mlp(cond_input)

        z_bt = z.unsqueeze(1).expand(b, self.time_steps, -1, h, w).reshape(b * self.time_steps, -1, h, w)
        cond_bt = cond_seq.reshape(b * self.time_steps, -1)

        for fno in self.fno_blocks:
            z_bt = fno(z_bt, cond_bt)

        return self.decoder(z_bt).view(b, self.time_steps, 2, h, w)


# ==========================================
# TRAINING LOGIC
# ==========================================
def compute_loss_terms(pred, gt, cfg: Config):
    def seg_l(p, g):
        l1 = F.smooth_l1_loss(p, g)
        grad = spatial_gradient_loss(p, g)
        return l1 + cfg.grad_weight * grad

    l_sf = (
        cfg.seg_t0_weight * seg_l(pred[:, 0:1, 0], gt[:, 0:1, 0])
        + cfg.seg_t1_20_weight * seg_l(pred[:, 1:21, 0], gt[:, 1:21, 0])
        + cfg.seg_t21_60_weight * seg_l(pred[:, 21:61, 0], gt[:, 21:61, 0])
    )
    l_vd = seg_l(pred[:, :, 1], gt[:, :, 1])
    total_loss = cfg.sf_weight * l_sf + cfg.vd_weight * l_vd
    return total_loss, l_sf, l_vd


def compute_all_metrics(pred, gt, cfg: Config):
    total_loss, l_sf, l_vd = compute_loss_terms(pred, gt, cfg)
    with torch.no_grad():
        metrics = {
            "loss": float(total_loss.item()),
            "sf_loss": float(l_sf.item()),
            "vd_loss": float(l_vd.item()),
            "sf_r2": float(torch_r2_score(pred[:, :, 0], gt[:, :, 0]).item()),
            "vd_r2": float(torch_r2_score(pred[:, :, 1], gt[:, :, 1]).item()),
            "sf_rmse": float(compute_rmse(pred[:, :, 0], gt[:, :, 0]).item()),
            "vd_rmse": float(compute_rmse(pred[:, :, 1], gt[:, :, 1]).item()),
        }
    return total_loss, metrics


def init_running_stats(metric_keys):
    return {
        key: {
            "count": 0,
            "mean": 0.0,
            "m2": 0.0,
        }
        for key in metric_keys
    }


def update_running_stats(running_stats, metrics):
    for key, value in metrics.items():
        stats = running_stats[key]
        stats["count"] += 1
        delta = float(value) - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = float(value) - stats["mean"]
        stats["m2"] += delta * delta2


def finalize_running_stats(running_stats):
    aggregated = {}
    for key, stats in running_stats.items():
        aggregated[key] = float(stats["mean"])
        if stats["count"] > 1:
            aggregated[f"{key}_std"] = float(np.sqrt(stats["m2"] / (stats["count"] - 1)))
        else:
            aggregated[f"{key}_std"] = 0.0
    return aggregated


def run_one_epoch(model, loader, optimizer, cfg: Config, device: torch.device, train: bool):
    global _CUDA_BATCH_REPORT_EMITTED

    model.train(train)
    running = {
        "loss": torch.zeros((), device=device),
        "sf_loss": torch.zeros((), device=device),
        "vd_loss": torch.zeros((), device=device),
    }
    num_batches = 0
    progress_interval = max(1, int(cfg.progress_interval))

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for x, d, inj, y in pbar:
        x = x.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        inj = inj.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred = model(x, d, inj)
            loss, l_sf, l_vd = compute_loss_terms(pred, y, cfg)
            if train and device.type == "cuda" and not _CUDA_BATCH_REPORT_EMITTED:
                cuda_idx = 0 if device.index is None else device.index
                print(
                    "[CUDA] First training batch on GPU | "
                    f"x={x.device} y={y.device} pred={pred.device} | "
                    f"allocated={torch.cuda.memory_allocated(cuda_idx) / (1024**2):.1f} MiB | "
                    f"reserved={torch.cuda.memory_reserved(cuda_idx) / (1024**2):.1f} MiB"
                )
                _CUDA_BATCH_REPORT_EMITTED = True
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        num_batches += 1
        with torch.no_grad():
            running["loss"] += loss.detach()
            running["sf_loss"] += l_sf.detach()
            running["vd_loss"] += l_vd.detach()

        if num_batches == 1 or num_batches % progress_interval == 0:
            pbar.set_postfix(
                loss=f"{(running['loss'] / num_batches).item():.4f}",
                sf_loss=f"{(running['sf_loss'] / num_batches).item():.4f}",
                vd_loss=f"{(running['vd_loss'] / num_batches).item():.4f}",
            )

    if num_batches == 0:
        return {key: 0.0 for key in running}

    return {key: float((value / num_batches).item()) for key, value in running.items()}


def evaluate_epoch(model, loader, cfg: Config, device: torch.device, calibration: dict):
    model.eval()
    running = {
        "loss": 0.0,
        "sf_loss": 0.0,
        "vd_loss": 0.0,
        "sf_r2": 0.0,
        "vd_r2": 0.0,
        "sf_rmse": 0.0,
        "vd_rmse": 0.0,
        "sf_uncertainty_mean": 0.0,
        "vd_uncertainty_mean": 0.0,
        "sf_confidence_mean": 0.0,
        "vd_confidence_mean": 0.0,
        "sf_uncertainty_p95": 0.0,
        "vd_uncertainty_p95": 0.0,
    }
    running_stats = init_running_stats(running.keys())
    num_batches = 0

    pbar = tqdm(loader, desc="eval", leave=False)
    with torch.no_grad():
        for x, d, inj, y in pbar:
            x = x.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)
            inj = inj.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred, pred_std = predict_with_uncertainty(model, x, d, inj, cfg.uncertainty_passes)
            loss, metrics = compute_all_metrics(pred, y, cfg)
            metrics.update(summarize_uncertainty(pred_std, calibration))

            num_batches += 1
            update_running_stats(running_stats, metrics)
            for key in running:
                running[key] += metrics[key]

            pbar.set_postfix(
                loss=f"{running['loss'] / num_batches:.4f}",
                sf_loss=f"{running['sf_loss'] / num_batches:.4f}",
                vd_loss=f"{running['vd_loss'] / num_batches:.4f}",
                sf_rmse=f"{running['sf_rmse'] / num_batches:.4f}",
                vd_rmse=f"{running['vd_rmse'] / num_batches:.4f}",
                sf_r2=f"{running['sf_r2'] / num_batches:.4f}",
                vd_r2=f"{running['vd_r2'] / num_batches:.4f}",
                sf_unc=f"{running['sf_uncertainty_mean'] / num_batches:.4f}",
                vd_unc=f"{running['vd_uncertainty_mean'] / num_batches:.4f}",
            )

    return finalize_running_stats(running_stats)


def build_datasets(cfg: Config):
    train_path = resolve_dir(cfg.train_dir, cfg.data_root, ("train",), "train")
    train_ds_full = DatasetLayers(train_path, max_layer=cfg.time_steps - 1)
    if len(train_ds_full) == 0:
        raise FileNotFoundError(f"No se encontraron datos en {train_path}")

    if cfg.overfit_sample_idx is not None:
        idx = int(cfg.overfit_sample_idx)
        if idx < 0 or idx >= len(train_ds_full):
            raise IndexError(
                f"overfit_sample_idx={idx} fuera de rango. Dataset size={len(train_ds_full)}"
            )
        train_ds = Subset(train_ds_full, [idx])
        val_ds = Subset(train_ds_full, [idx])
        return train_ds, val_ds, train_path, train_path

    if cfg.val_dir is not None:
        val_path = resolve_dir(cfg.val_dir, cfg.data_root, (), "validacion")
        val_ds = DatasetLayers(val_path, max_layer=cfg.time_steps - 1)
        return train_ds_full, val_ds, train_path, val_path

    default_test = Path(cfg.data_root) / "test"
    if default_test.exists() and default_test.is_dir():
        val_ds = DatasetLayers(default_test, max_layer=cfg.time_steps - 1)
        return train_ds_full, val_ds, train_path, default_test

    n_train = max(1, int(0.9 * len(train_ds_full)))
    n_val = len(train_ds_full) - n_train
    if n_val == 0:
        n_train = len(train_ds_full) - 1
        n_val = 1
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(train_ds_full, [n_train, n_val], generator=gen)
    return train_ds, val_ds, train_path, train_path


# ==========================================
# VISUALIZATION
# ==========================================
def load_or_create_uncertainty_calibration(path: Path, model, val_loader, cfg: Config, device: torch.device):
    if not model_has_dropout(model):
        calibration = default_uncertainty_calibration()
        save_json(path, calibration)
        print("[UNCERTAINTY] Modelo sin Dropout: calibracion desactivada para no frenar el entrenamiento.")
        return calibration

    if path.exists():
        return load_json(path, default={})

    calibration = calibrate_uncertainty(model, val_loader, cfg, device)
    save_json(path, calibration)
    return calibration


def save_history_plots(history, out_dir):
    if not history:
        return

    path = Path(out_dir)
    ensure_dir(path)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="Train loss", linewidth=2.0)
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history], label="Val loss", linewidth=2.0)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(epochs, [row["train_sf_rmse"] for row in history], label="Train SF RMSE", linewidth=2.0)
    axes[0, 1].plot(epochs, [row["val_sf_rmse"] for row in history], label="Val SF RMSE", linewidth=2.0)
    axes[0, 1].plot(epochs, [row["train_vd_rmse"] for row in history], label="Train VD RMSE", linewidth=2.0)
    axes[0, 1].plot(epochs, [row["val_vd_rmse"] for row in history], label="Val VD RMSE", linewidth=2.0)
    axes[0, 1].set_title("RMSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(epochs, [row["train_sf_r2"] for row in history], label="Train SF R2", linewidth=2.0)
    axes[1, 0].plot(epochs, [row["val_sf_r2"] for row in history], label="Val SF R2", linewidth=2.0)
    axes[1, 0].plot(epochs, [row["train_vd_r2"] for row in history], label="Train VD R2", linewidth=2.0)
    axes[1, 0].plot(epochs, [row["val_vd_r2"] for row in history], label="Val VD R2", linewidth=2.0)
    axes[1, 0].set_title("R2")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(
        epochs,
        [row.get("val_sf_uncertainty_mean", 0.0) for row in history],
        label="Val SF uncertainty",
        linewidth=2.0,
    )
    axes[1, 1].plot(
        epochs,
        [row.get("val_vd_uncertainty_mean", 0.0) for row in history],
        label="Val VD uncertainty",
        linewidth=2.0,
    )
    axes[1, 1].set_title("Average prediction uncertainty")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("0-1")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(path / "training_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        epochs,
        [row.get("val_sf_confidence_mean", 0.0) for row in history],
        label="Val SF confidence",
        linewidth=2.0,
    )
    axes[0].plot(
        epochs,
        [row.get("val_vd_confidence_mean", 0.0) for row in history],
        label="Val VD confidence",
        linewidth=2.0,
    )
    axes[0].set_title("Confidence")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("0-1")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        epochs,
        [row.get("val_sf_uncertainty_p95", 0.0) for row in history],
        label="Val SF uncertainty p95",
        linewidth=2.0,
    )
    axes[1].plot(
        epochs,
        [row.get("val_vd_uncertainty_p95", 0.0) for row in history],
        label="Val VD uncertainty p95",
        linewidth=2.0,
    )
    axes[1].set_title("Uncertainty p95")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("0-1")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(path / "uncertainty_curves.png")
    plt.close(fig)


def save_epoch_visuals(model, dataset, epoch, out_dir, device, cfg: Config, calibration: dict):
    if dataset is None or len(dataset) == 0:
        return

    model.eval()
    path = Path(out_dir) / "visuals"
    ensure_dir(path)

    with torch.no_grad():
        if isinstance(dataset, Subset):
            rel_idx = random.randrange(len(dataset.indices))
            base_idx = int(dataset.indices[rel_idx])
            base_dataset = dataset.dataset
        else:
            base_idx = random.randrange(len(dataset))
            base_dataset = dataset

        x, d, inj, y = base_dataset[base_idx]
        meta = base_dataset.samples[base_idx]
        pred, pred_std = predict_with_uncertainty(
            model,
            x.unsqueeze(0).to(device),
            d.unsqueeze(0).to(device),
            inj.unsqueeze(0).to(device),
            cfg.uncertainty_passes,
        )
        pred = pred[0].cpu()
        pred_std = pred_std[0].cpu()
        uncertainty_map = build_uncertainty_map(pred_std.unsqueeze(0), calibration)[0]
        confidence_map = 1.0 - uncertainty_map

        times = [0, min(30, y.shape[0] - 1), y.shape[0] - 1]
        times = list(dict.fromkeys(times))
        fig, axes = plt.subplots(8, len(times), figsize=(6 * len(times), 26))
        if len(times) == 1:
            axes = np.expand_dims(axes, axis=1)

        for col, t in enumerate(times):
            for row, data, title, cmap, vmax in [
                (0, y, "GT SF", "viridis", 1.0),
                (1, pred, "Pred SF", "viridis", 1.0),
                (2, uncertainty_map, "SF uncertainty", "inferno", 1.0),
                (3, confidence_map, "SF confidence", "viridis", 1.0),
            ]:
                im = axes[row, col].imshow(data[t, 0], cmap=cmap, vmin=0, vmax=vmax)
                axes[row, col].set_title(f"{title} t={t}")
                divider = make_axes_locatable(axes[row, col])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

            vmax_vd = max(y[:, 1].max().item(), pred[:, 1].max().item()) + 1e-6
            for row, data, title, cmap in [
                (4, y, "GT VD", "plasma"),
                (5, pred, "Pred VD", "plasma"),
            ]:
                im = axes[row, col].imshow(data[t, 1], cmap=cmap, vmin=0, vmax=vmax_vd)
                axes[row, col].set_title(f"{title} t={t}")
                divider = make_axes_locatable(axes[row, col])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

            im = axes[6, col].imshow(uncertainty_map[t, 1], cmap="inferno", vmin=0, vmax=1.0)
            axes[6, col].set_title(f"VD uncertainty t={t}")
            divider = make_axes_locatable(axes[6, col])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

            im = axes[7, col].imshow(confidence_map[t, 1], cmap="viridis", vmin=0, vmax=1.0)
            axes[7, col].set_title(f"VD confidence t={t}")
            divider = make_axes_locatable(axes[7, col])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Epoch {epoch:03d} | case={meta['case']} | layer={meta['k']}", fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.savefig(path / f"epoch_{epoch:03d}.png")
        plt.close(fig)

        input_fig, input_axes = plt.subplots(2, 3, figsize=(16, 10))
        input_panels = [
            ("AFI", x[0], "viridis"),
            ("COH", x[1], "viridis"),
            ("PERM", x[2], "magma"),
            ("PORO", x[3], "cividis"),
        ]

        for ax, (title, data, cmap) in zip(input_axes.flatten(), input_panels):
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            input_fig.colorbar(im, cax=cax)

        inj_ax = input_axes[1, 1]
        t = np.arange(inj.shape[0])
        inj_np = inj.cpu().numpy()
        inj_ax.plot(t, inj_np[:, 0], label="Inj 1", linewidth=2.0)
        inj_ax.plot(t, inj_np[:, 1], label="Inj 2", linewidth=2.0)
        inj_ax.set_title("Real Injection Input")
        inj_ax.set_xlabel("Time")
        inj_ax.set_ylabel("Normalized value")
        inj_ax.grid(alpha=0.3)
        inj_ax.legend(loc="best")

        meta_ax = input_axes[1, 2]
        meta_ax.axis("off")
        meta_ax.text(
            0.0,
            1.0,
            "\n".join(
                [
                    f"Epoch: {epoch:03d}",
                    f"Case: {meta['case']}",
                    f"Layer: {meta['k']}",
                    f"Depth scalar: {float(d.item()):.4f}",
                    f"Input shape: {tuple(x.shape)}",
                    f"Injection shape: {tuple(inj.shape)}",
                    f"Inj max abs: {float(inj.abs().max().item()):.4f}",
                    f"Mean SF uncertainty: {float(uncertainty_map[:, 0].mean().item()):.4f}",
                    f"Mean VD uncertainty: {float(uncertainty_map[:, 1].mean().item()):.4f}",
                    f"Mean SF confidence: {float(confidence_map[:, 0].mean().item()):.4f}",
                    f"Mean VD confidence: {float(confidence_map[:, 1].mean().item()):.4f}",
                ]
            ),
            va="top",
            ha="left",
            fontsize=11,
        )

        input_fig.suptitle(
            f"Epoch {epoch:03d} Inputs | case={meta['case']} | layer={meta['k']}",
            fontsize=14,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(path / f"epoch_{epoch:03d}_inputs.png")
        plt.close(input_fig)

        temporal_fig, temporal_axes = plt.subplots(1, 2, figsize=(14, 5))
        sf_unc_time = uncertainty_map[:, 0].mean(dim=(1, 2)).numpy()
        vd_unc_time = uncertainty_map[:, 1].mean(dim=(1, 2)).numpy()
        temporal_axes[0].plot(np.arange(len(sf_unc_time)), sf_unc_time, linewidth=2.0, color="tab:blue")
        temporal_axes[0].set_title("SF mean uncertainty by time")
        temporal_axes[0].set_xlabel("Time")
        temporal_axes[0].set_ylabel("0-1")
        temporal_axes[0].grid(alpha=0.3)

        temporal_axes[1].plot(np.arange(len(vd_unc_time)), vd_unc_time, linewidth=2.0, color="tab:orange")
        temporal_axes[1].set_title("VD mean uncertainty by time")
        temporal_axes[1].set_xlabel("Time")
        temporal_axes[1].set_ylabel("0-1")
        temporal_axes[1].grid(alpha=0.3)

        temporal_fig.suptitle(
            f"Epoch {epoch:03d} uncertainty timeline | case={meta['case']} | layer={meta['k']}",
            fontsize=14,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(path / f"epoch_{epoch:03d}_uncertainty_timeline.png")
        plt.close(temporal_fig)


# ==========================================
# CLI
# ==========================================
def str_to_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y", "si", "sí"}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--train-dir", default=None)
    p.add_argument("--val-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--overfit-sample-idx", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=None)
    p.add_argument("--persistent-workers", type=str_to_bool, default=None)
    p.add_argument("--progress-interval", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--pause-hour", type=int, default=None)
    p.add_argument("--auto-resume", type=str_to_bool, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--early-stopping-min-delta", type=float, default=None)
    p.add_argument("--uncertainty-passes", type=int, default=None)
    p.add_argument("--save-epoch-pngs", type=str_to_bool, default=None)
    return p


# ==========================================
# MAIN
# ==========================================
def main():
    args = build_parser().parse_args()
    cfg = copy.deepcopy(CFG)
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    print(f"[PYTHON] {sys.executable}")
    print(f"[DEVICE] Using {describe_device(device)}")

    out_path = Path(cfg.output_dir)
    ensure_dir(out_path)

    ckpt_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else (out_path / "checkpoints")
    ensure_dir(ckpt_dir)

    latest_ckpt = ckpt_dir / "latest.pt"
    best_ckpt = ckpt_dir / "best.pt"
    history_path = out_path / "metrics_history.json"
    calibration_path = out_path / "uncertainty_calibration.json"

    train_ds, val_ds, train_path, val_path = build_datasets(cfg)
    run_signature = build_run_signature(cfg, train_path, val_path)

    train_dl, resolved_num_workers = build_loader(
        train_ds,
        cfg,
        device,
        shuffle=(cfg.overfit_sample_idx is None),
    )
    val_dl, _ = build_loader(val_ds, cfg, device, shuffle=False)

    model = PhysicalFNOArchitecture(
        time_steps=cfg.time_steps,
        in_c=5,
        h_dim=cfg.hidden_dim,
        modes=cfg.spectral_modes,
    ).to(device)
    assert_model_on_device(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    resumed_from = None

    if cfg.auto_resume:
        resume_candidates = [latest_ckpt]
        if best_ckpt != latest_ckpt and best_ckpt.exists():
            resume_candidates.append(best_ckpt)

        for candidate in resume_candidates:
            if not candidate.exists():
                continue

            start_epoch, best_val_loss, last_metrics, resumed, reasons, resume_path = try_resume_training(
                candidate,
                model,
                optimizer,
                device,
                run_signature,
            )
            if resumed:
                resumed_from = resume_path
                history = load_json(history_path, default=[])
                print(
                    f"[RESUME] Loaded {resume_path.name}. "
                    f"Continuing from epoch {start_epoch} (last complete: {start_epoch - 1})"
                )
                if last_metrics is not None:
                    print(
                        f"[RESUME] Last val_loss={last_metrics.get('val_loss', 'n/a')} | "
                        f"best val_loss={best_val_loss:.6f}"
                    )
                break

            print(f"[RESUME SKIP] {candidate.name} was not resumed:")
            for reason in reasons:
                print(f"  - {reason}")

        if resumed_from is None:
            print("[START] Starting a fresh run for the current configuration")
    else:
        print("[START] Training from scratch")

    if start_epoch > cfg.epochs:
        print(f"Training already reached epoch {cfg.epochs}.")
        return

    save_json(
        out_path / "config.json",
        {
            **asdict(cfg),
            "train_dir": str(train_path),
            "val_dir": str(val_path),
            "checkpoint_dir": str(ckpt_dir),
            "resolved_device": describe_device(device),
            "resolved_num_workers": resolved_num_workers,
            "run_signature": run_signature,
        },
    )

    print(f"Train dir: {train_path}")
    print(f"Val dir: {val_path}")
    print(f"Output dir: {out_path}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(
        "DataLoader: "
        f"batch_size={cfg.batch_size} | "
        f"num_workers={resolved_num_workers} | "
        f"pin_memory={device.type == 'cuda'} | "
        f"prefetch_factor={cfg.prefetch_factor if resolved_num_workers > 0 else 'n/a'} | "
        f"persistent_workers={bool(cfg.persistent_workers) if resolved_num_workers > 0 else 'n/a'}"
    )
    if cfg.overfit_sample_idx is not None:
        print(f"OVERFIT MODE ACTIVE | sample_idx={cfg.overfit_sample_idx}")
    print(f"Device: {describe_device(device)}")
    if device.type == "cuda":
        cuda_idx = 0 if device.index is None else device.index
        print(
            "CUDA memory after model load: "
            f"allocated={torch.cuda.memory_allocated(cuda_idx) / (1024**2):.1f} MiB | "
            f"reserved={torch.cuda.memory_reserved(cuda_idx) / (1024**2):.1f} MiB"
        )
    print(f"Parameters: {count_parameters(model):,}")
    print(
        "[EARLY STOPPING] "
        f"patience={cfg.early_stopping_patience} | "
        f"min_delta={cfg.early_stopping_min_delta}"
    )

    next_pause_dt = get_next_pause_datetime(cfg.pause_hour)
    print(f"[AUTO-PAUSE] Will stop after an epoch once it reaches: {next_pause_dt:%Y-%m-%d %H:%M:%S}")
    calibration = load_or_create_uncertainty_calibration(calibration_path, model, val_dl, cfg, device)

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = run_one_epoch(model, train_dl, optimizer, cfg, device, train=True)
        calibration = calibrate_uncertainty(model, val_dl, cfg, device)
        save_json(calibration_path, calibration)
        train_eval_metrics = evaluate_epoch(model, train_dl, cfg, device, calibration)
        val_metrics = evaluate_epoch(model, val_dl, cfg, device, calibration)

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_eval_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        save_json(history_path, history)
        save_history_plots(history, out_path)

        print(
            f"E{epoch:03d} | "
            f"train_loss={train_metrics['loss']:.5f} | "
            f"val_loss={row['val_loss']:.5f} | "
            f"val_sf_rmse={row['val_sf_rmse']:.5f} | "
            f"val_vd_rmse={row['val_vd_rmse']:.5f} | "
            f"val_sf_r2={row['val_sf_r2']:.4f} | "
            f"val_vd_r2={row['val_vd_r2']:.4f} | "
            f"val_sf_unc={row['val_sf_uncertainty_mean']:.4f} | "
            f"val_vd_unc={row['val_vd_uncertainty_mean']:.4f} | "
            f"val_sf_conf={row['val_sf_confidence_mean']:.4f} | "
            f"val_vd_conf={row['val_vd_confidence_mean']:.4f}"
        )

        if cfg.save_epoch_pngs:
            save_epoch_visuals(model, val_ds, epoch, out_path, device, cfg, calibration)

        if row["val_loss"] < (best_val_loss - cfg.early_stopping_min_delta):
            best_val_loss = row["val_loss"]
            epochs_without_improvement = 0
            save_training_checkpoint(
                best_ckpt,
                model,
                optimizer,
                cfg,
                epoch,
                best_val_loss,
                row,
                run_signature,
            )
            print(f"  new best val_loss={best_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            print(
                "  no improvement in val_loss "
                f"({epochs_without_improvement}/{cfg.early_stopping_patience})"
            )

        save_training_checkpoint(
            latest_ckpt,
            model,
            optimizer,
            cfg,
            epoch,
            best_val_loss,
            row,
            run_signature,
        )

        if datetime.now() >= next_pause_dt:
            print(
                f"\n[AUTO-PAUSE] Reached {cfg.pause_hour}:00. "
                f"Checkpoint saved in {latest_ckpt.resolve()}"
            )
            print("Run the same command again and it will continue from the next epoch.")
            return

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                "\n[EARLY STOPPING] "
                f"Stopped at epoch {epoch} after "
                f"{cfg.early_stopping_patience} epochs without improving val_loss by at least "
                f"{cfg.early_stopping_min_delta}."
            )
            print(f"Best val_loss={best_val_loss:.6f}")
            return

    print(f"\nTraining finished. Best val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()
