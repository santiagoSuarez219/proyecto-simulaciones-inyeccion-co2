import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from modelo_itm.config import Config
from modelo_itm.data.loaders import build_datasets, build_loader
from modelo_itm.inference.uncertainty import (
    calibrate_uncertainty,
    load_or_create_uncertainty_calibration,
    predict_with_uncertainty,
    summarize_uncertainty,
)
from modelo_itm.models.fno import PhysicalFNOArchitecture
from modelo_itm.training.checkpoint import (
    build_run_signature,
    save_training_checkpoint,
    try_resume_training,
)
from modelo_itm.training.metrics import (
    compute_all_metrics,
    compute_rmse,
    finalize_running_stats,
    init_running_stats,
    torch_r2_score,
    update_running_stats,
)
from modelo_itm.training.losses import compute_loss_terms
from modelo_itm.utils import resolve_device, seed_everything
from modelo_itm.utils.io import ensure_dir, load_json, save_json
from modelo_itm.utils.time import get_next_pause_datetime
from modelo_itm.visualization import save_epoch_visuals, save_history_plots

_CUDA_BATCH_REPORT_EMITTED = False


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


def main(cfg: Config):
    device = resolve_device(cfg.device)
    print(f"[DEVICE] {device}")
    seed_everything(cfg.seed)

    train_ds, val_ds, train_path, val_path = build_datasets(cfg)
    train_loader, train_num_workers = build_loader(train_ds, cfg, device, shuffle=True)
    val_loader, val_num_workers = build_loader(val_ds, cfg, device, shuffle=False)

    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} | workers train={train_num_workers} val={val_num_workers}")

    model = PhysicalFNOArchitecture(
        time_steps=cfg.time_steps,
        in_c=5,
        h_dim=cfg.hidden_dim,
        modes=cfg.spectral_modes,
        cond_dim=128,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    ckpt_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else out_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    run_signature = build_run_signature(cfg, train_path, val_path)
    save_json(out_dir / "config.json", {k: str(v) for k, v in run_signature.items()})

    history = load_json(out_dir / "metrics_history.json", default=[])
    calibration_path = out_dir / "uncertainty_calibration.json"
    best_ckpt_path = ckpt_dir / "best.pt"
    latest_ckpt_path = ckpt_dir / "latest.pt"

    start_epoch = 1
    best_val_loss = float("inf")

    if cfg.auto_resume:
        start_epoch, best_val_loss, _, resumed, reasons, _ = try_resume_training(
            latest_ckpt_path, model, optimizer, device, run_signature
        )
        if resumed:
            print(f"[RESUME] Continuando desde epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
        else:
            print(f"[RESUME FAILED] {' | '.join(reasons)}")

    calibration = load_or_create_uncertainty_calibration(calibration_path, model, val_loader, cfg, device)

    early_stopping_counter = 0
    pause_datetime = get_next_pause_datetime(cfg.pause_hour)
    print(f"[SCHEDULE] Pausa programada a {pause_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, cfg, device, train=True)
        val_metrics = evaluate_epoch(model, val_loader, cfg, device, calibration)

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss", 0.0),
            "train_sf_loss": train_metrics.get("sf_loss", 0.0),
            "train_vd_loss": train_metrics.get("vd_loss", 0.0),
            "val_loss": val_metrics.get("loss", 0.0),
            "val_sf_loss": val_metrics.get("sf_loss", 0.0),
            "val_vd_loss": val_metrics.get("vd_loss", 0.0),
            "train_sf_r2": val_metrics.get("sf_r2", 0.0),
            "train_vd_r2": val_metrics.get("vd_r2", 0.0),
            "val_sf_r2": val_metrics.get("sf_r2", 0.0),
            "val_vd_r2": val_metrics.get("vd_r2", 0.0),
            "train_sf_rmse": train_metrics.get("sf_loss", 0.0),
            "train_vd_rmse": train_metrics.get("vd_loss", 0.0),
            "val_sf_rmse": val_metrics.get("sf_rmse", 0.0),
            "val_vd_rmse": val_metrics.get("vd_rmse", 0.0),
        }
        history_row.update({k: v for k, v in val_metrics.items() if k.startswith(("sf_", "vd_", "val_"))})
        history.append(history_row)
        save_json(out_dir / "metrics_history.json", history)

        save_training_checkpoint(
            latest_ckpt_path,
            model,
            optimizer,
            cfg,
            epoch,
            best_val_loss,
            history_row,
            run_signature,
        )

        val_loss = val_metrics.get("loss", float("inf"))
        if val_loss < best_val_loss - cfg.early_stopping_min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0
            save_training_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                cfg,
                epoch,
                best_val_loss,
                history_row,
                run_signature,
            )
            print(f"[EPOCH {epoch:04d}] New best val_loss={best_val_loss:.6f}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= cfg.early_stopping_patience:
                print(f"[EARLY STOP] No improvement for {cfg.early_stopping_patience} epochs. Deteniendo entrenamiento.")
                break

        if cfg.save_epoch_pngs:
            save_epoch_visuals(model, val_ds, epoch, out_dir, device, cfg, calibration)

        if len(history) % 5 == 0:
            save_history_plots(history, out_dir)

    save_history_plots(history, out_dir)
    print(f"[TRAIN COMPLETE] Entrenamiento finalizado. best_val_loss={best_val_loss:.6f}")
