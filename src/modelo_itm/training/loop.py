from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from modelo_itm.config import Config
from modelo_itm.data.loaders import build_datasets, build_loader
from modelo_itm.inference.uncertainty import (
    calibrate_uncertainty,
    load_or_create_uncertainty_calibration,
    model_has_dropout,
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
    count_parameters,
    finalize_global_regression_metrics,
    finalize_running_stats,
    init_global_regression_accumulators,
    init_running_stats,
    update_global_regression_accumulators,
    update_running_stats,
)
from modelo_itm.training.losses import compute_loss_terms
from modelo_itm.utils import describe_device, resolve_device, seed_everything
from modelo_itm.utils.io import ensure_dir, load_json, save_json
from modelo_itm.utils.time import get_next_pause_datetime
from modelo_itm.visualization import save_epoch_visuals, save_history_plots

_CUDA_BATCH_REPORT_EMITTED = False


def run_one_epoch(model, loader, optimizer, cfg: Config, device: torch.device, train: bool):
    """Acumula tambien sf_r2/vd_r2/sf_rmse/vd_rmse globalmente durante el propio
    paso de entrenamiento (misma acumulacion global de C3), aprovechando el pred/y
    que ya se calculan para la loss — evita el forward extra sobre todo train que
    evaluate_epoch(train_loader) hacia cada epoca (A3)."""
    global _CUDA_BATCH_REPORT_EMITTED

    model.train(train)
    running = {
        "loss": torch.zeros((), device=device),
        "sf_loss": torch.zeros((), device=device),
        "vd_loss": torch.zeros((), device=device),
    }
    regression_accumulators = init_global_regression_accumulators(("sf", "vd"))
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
            update_global_regression_accumulators(regression_accumulators, "sf", pred[:, :, 0], y[:, :, 0])
            update_global_regression_accumulators(regression_accumulators, "vd", pred[:, :, 1], y[:, :, 1])

        if num_batches == 1 or num_batches % progress_interval == 0:
            pbar.set_postfix(
                loss=f"{(running['loss'] / num_batches).item():.4f}",
                sf_loss=f"{(running['sf_loss'] / num_batches).item():.4f}",
                vd_loss=f"{(running['vd_loss'] / num_batches).item():.4f}",
            )

    if num_batches == 0:
        result = {key: 0.0 for key in running}
    else:
        result = {key: float((value / num_batches).item()) for key, value in running.items()}
    result.update(finalize_global_regression_metrics(regression_accumulators))
    return result


def evaluate_epoch(model, loader, cfg: Config, device: torch.device, calibration: dict):
    """sf_r2/vd_r2/sf_rmse/vd_rmse se calculan una sola vez al final sobre el dataset
    completo (acumulación global de SS_res/SS_tot y suma de errores cuadrados). El resto
    de las métricas (loss, incertidumbre) sí son promediables por batch."""
    model.eval()
    running_keys = (
        "loss",
        "sf_loss",
        "vd_loss",
        "sf_uncertainty_mean",
        "vd_uncertainty_mean",
        "sf_confidence_mean",
        "vd_confidence_mean",
        "sf_uncertainty_p95",
        "vd_uncertainty_p95",
    )
    running_stats = init_running_stats(running_keys)
    regression_accumulators = init_global_regression_accumulators(("sf", "vd"))
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

            update_global_regression_accumulators(regression_accumulators, "sf", pred[:, :, 0], y[:, :, 0])
            update_global_regression_accumulators(regression_accumulators, "vd", pred[:, :, 1], y[:, :, 1])

            num_batches += 1
            update_running_stats(running_stats, metrics)

            partial_regression = finalize_global_regression_metrics(regression_accumulators)
            pbar.set_postfix(
                loss=f"{running_stats['loss']['mean']:.4f}",
                sf_loss=f"{running_stats['sf_loss']['mean']:.4f}",
                vd_loss=f"{running_stats['vd_loss']['mean']:.4f}",
                sf_rmse=f"{partial_regression['sf_rmse']:.4f}",
                vd_rmse=f"{partial_regression['vd_rmse']:.4f}",
                sf_r2=f"{partial_regression['sf_r2']:.4f}",
                vd_r2=f"{partial_regression['vd_r2']:.4f}",
                sf_unc=f"{running_stats['sf_uncertainty_mean']['mean']:.4f}",
                vd_unc=f"{running_stats['vd_uncertainty_mean']['mean']:.4f}",
            )

    result = finalize_running_stats(running_stats)
    result.update(finalize_global_regression_metrics(regression_accumulators))
    return result


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
    save_json(
        out_dir / "config.json",
        {
            **{k: str(v) for k, v in asdict(cfg).items()},
            "train_dir": str(train_path),
            "val_dir": str(val_path),
            "checkpoint_dir": str(ckpt_dir),
            "resolved_device": describe_device(device),
            "resolved_num_workers": train_num_workers,
            "run_signature": {k: str(v) for k, v in run_signature.items()},
        },
    )
    print(f"[MODEL] Parametros entrenables: {count_parameters(model):,}")

    history_path = out_dir / "metrics_history.json"
    calibration_path = out_dir / "uncertainty_calibration.json"
    best_ckpt_path = ckpt_dir / "best.pt"
    latest_ckpt_path = ckpt_dir / "latest.pt"

    start_epoch = 1
    best_val_loss = float("inf")
    history = []
    resumed_from = None

    if cfg.auto_resume:
        resume_candidates = [latest_ckpt_path]
        if best_ckpt_path != latest_ckpt_path and best_ckpt_path.exists():
            resume_candidates.append(best_ckpt_path)

        for candidate in resume_candidates:
            if not candidate.exists():
                continue

            start_epoch, best_val_loss, last_metrics, resumed, reasons, resume_path = try_resume_training(
                candidate, model, optimizer, device, run_signature
            )
            if resumed:
                resumed_from = resume_path
                history = load_json(history_path, default=[])
                print(
                    f"[RESUME] Cargado {resume_path.name}. "
                    f"Continuando desde epoch {start_epoch} (ultima completa: {start_epoch - 1})"
                )
                if last_metrics is not None:
                    print(
                        f"[RESUME] Ultimo val_loss={last_metrics.get('val_loss', 'n/a')} | "
                        f"best val_loss={best_val_loss:.6f}"
                    )
                break

            print(f"[RESUME SKIP] {candidate.name} no se pudo reanudar:")
            for reason in reasons:
                print(f"  - {reason}")

        if resumed_from is None:
            print("[START] Iniciando una corrida nueva para la configuracion actual")
    else:
        print("[START] Entrenando desde cero")

    if start_epoch > cfg.epochs:
        print(f"El entrenamiento ya alcanzo la epoch {cfg.epochs}.")
        return

    calibration = load_or_create_uncertainty_calibration(calibration_path, model, val_loader, cfg, device)

    early_stopping_counter = 0
    pause_datetime = get_next_pause_datetime(cfg.pause_hour)
    print(f"[SCHEDULE] Pausa programada a {pause_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        # train_metrics ya incluye sf_r2/vd_r2/sf_rmse/vd_rmse (acumulados globalmente
        # durante el propio paso de entrenamiento) — evita el forward extra sobre todo
        # train que antes hacia evaluate_epoch(train_loader) cada epoca (A3).
        train_metrics = run_one_epoch(model, train_loader, optimizer, cfg, device, train=True)

        # Sin dropout (model_has_dropout=False) la calibracion es siempre el mismo
        # default trivial; recalibrar y reescribir el JSON cada epoca no aporta nada
        # (A3, ligado a C2). Solo se recalibra cuando la feature de incertidumbre
        # esta realmente activa.
        if model_has_dropout(model):
            calibration = calibrate_uncertainty(model, val_loader, cfg, device)
            save_json(calibration_path, calibration)

        val_metrics = evaluate_epoch(model, val_loader, cfg, device, calibration)

        history_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(history_row)
        save_json(history_path, history)
        save_history_plots(history, out_dir)

        print(
            f"E{epoch:03d} | "
            f"train_loss={train_metrics['loss']:.5f} | "
            f"val_loss={history_row['val_loss']:.5f} | "
            f"val_sf_rmse={history_row['val_sf_rmse']:.5f} | "
            f"val_vd_rmse={history_row['val_vd_rmse']:.5f} | "
            f"val_sf_r2={history_row['val_sf_r2']:.4f} | "
            f"val_vd_r2={history_row['val_vd_r2']:.4f} | "
            f"val_sf_unc={history_row['val_sf_uncertainty_mean']:.4f} | "
            f"val_vd_unc={history_row['val_vd_uncertainty_mean']:.4f} | "
            f"val_sf_conf={history_row['val_sf_confidence_mean']:.4f} | "
            f"val_vd_conf={history_row['val_vd_confidence_mean']:.4f}"
        )

        if cfg.save_epoch_pngs:
            save_epoch_visuals(model, val_ds, epoch, out_dir, device, cfg, calibration)

        val_loss = history_row["val_loss"]
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
            print(f"  nuevo best val_loss={best_val_loss:.6f}")
        else:
            early_stopping_counter += 1
            print(
                "  sin mejora en val_loss "
                f"({early_stopping_counter}/{cfg.early_stopping_patience})"
            )

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

        if datetime.now() >= pause_datetime:
            print(
                f"\n[AUTO-PAUSE] Se alcanzo la hora de pausa ({cfg.pause_hour}:00). "
                f"Checkpoint guardado en {latest_ckpt_path.resolve()}"
            )
            print("Ejecuta el mismo comando de nuevo y continuara desde la siguiente epoch.")
            return

        if early_stopping_counter >= cfg.early_stopping_patience:
            print(
                "\n[EARLY STOPPING] "
                f"Detenido en epoch {epoch} tras "
                f"{cfg.early_stopping_patience} epochs sin mejorar val_loss en al menos "
                f"{cfg.early_stopping_min_delta}."
            )
            print(f"Best val_loss={best_val_loss:.6f}")
            return

    print(f"\n[TRAIN COMPLETE] Entrenamiento finalizado. best_val_loss={best_val_loss:.6f}")
