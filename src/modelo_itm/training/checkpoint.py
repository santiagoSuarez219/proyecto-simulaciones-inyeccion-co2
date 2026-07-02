import os
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from modelo_itm.config import Config
from modelo_itm.utils.io import save_json


def build_run_signature(cfg: Config, train_path: Path, val_path: Path):
    return {
        "train_dir": str(train_path.resolve()),
        "val_dir": str(val_path.resolve()),
        "time_steps": int(cfg.time_steps),
        "hidden_dim": int(cfg.hidden_dim),
        "spectral_modes": int(cfg.spectral_modes),
        "dropout_p": float(cfg.dropout_p),
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
    scheduler=None,
):
    ckpt = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
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
    scheduler=None,
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

    if scheduler is not None:
        scheduler_state = ckpt.get("scheduler_state_dict")
        if scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except (RuntimeError, KeyError, ValueError) as exc:
                return 1, float("inf"), None, False, [f"estado de scheduler invalido en {ckpt_path.name}: {exc}"], None
        # Si el checkpoint no tiene estado de scheduler (guardado antes de M1, o
        # sin scheduler activo en esa corrida), el scheduler actual arranca desde
        # su estado inicial — no aborta el resume por esto.

    start_epoch = int(ckpt["epoch"]) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    last_metrics = ckpt.get("metrics")
    return start_epoch, best_val_loss, last_metrics, True, [], ckpt_path
