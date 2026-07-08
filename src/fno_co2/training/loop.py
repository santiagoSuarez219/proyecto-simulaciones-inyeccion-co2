from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from fno_co2.config import Config
from fno_co2.data.loaders import build_datasets, build_loader
from fno_co2.inference.uncertainty import (
    calibrate_uncertainty,
    default_uncertainty_calibration,
    model_has_dropout,
    predict_with_uncertainty,
    summarize_uncertainty,
)
from fno_co2.models.fno import PhysicalFNOArchitecture
from fno_co2.training.checkpoint import (
    build_run_signature,
    save_training_checkpoint,
    try_resume_training,
)
from fno_co2.training.metrics import (
    compute_all_metrics,
    count_parameters,
    finalize_global_regression_metrics,
    finalize_running_stats,
    init_global_regression_accumulators,
    init_running_stats,
    update_global_regression_accumulators,
    update_running_stats,
)
from fno_co2.training.losses import compute_loss_terms
from fno_co2.training.optim import build_param_groups, build_scheduler
from fno_co2.utils import EmitOnce, describe_device, get_logger, resolve_device, seed_everything
from fno_co2.utils.io import ensure_dir, load_json, save_json
from fno_co2.utils.time import get_next_pause_datetime
from fno_co2.visualization import save_epoch_visuals, save_history_plots

logger = get_logger(__name__)

_emit_once = EmitOnce()

# Dtype de autocast para AMP (M2). Se usa bfloat16 en vez de float16 porque el modelo
# tiene parametros espectrales ComplexFloat (FiLMSpectralBlock): con float16 haria falta
# un GradScaler, cuyo unscale_ NO soporta gradientes complejos
# ("_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'ComplexFloat'").
# bfloat16 tiene el mismo rango dinamico que float32, asi que no requiere loss scaling y
# evita por completo ese path. La FFT/multiplicacion espectral ya se fuerza a float32
# dentro de FiLMSpectralBlock (autocast(enabled=False)), independiente de este dtype.
_AMP_DTYPE = torch.bfloat16


def run_one_epoch(model, loader, optimizer, cfg: Config, device: torch.device, train: bool, scaler=None):
    """Acumula tambien sf_r2/vd_r2/sf_rmse/vd_rmse globalmente durante el propio
    paso de entrenamiento (misma acumulacion global de C3), aprovechando el pred/y
    que ya se calculan para la loss — evita el forward extra sobre todo train que
    evaluate_epoch(train_loader) hacia cada epoca (A3).

    Con cfg.use_amp=True (y device.type=="cuda") el forward+loss corren bajo
    torch.autocast en bfloat16 (_AMP_DTYPE) — sin GradScaler, porque bf16 no necesita
    loss scaling y el modelo tiene params complejos incompatibles con unscale_ (ver
    _AMP_DTYPE). El scaler se conserva como no-op transparente (siempre enabled=False)
    para no alterar el flujo de guardas M6."""
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu", enabled=False)
    autocast_enabled = bool(cfg.use_amp) and device.type == "cuda"

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
            with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=_AMP_DTYPE):
                pred = model(x, d, inj)
                loss, l_sf, l_vd = compute_loss_terms(pred, y, cfg)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[NaN/Inf GUARD] Loss no finita en batch {num_batches + 1} "
                    f"(train={train}): loss={loss.item()}, sf_loss={l_sf.item()}, "
                    f"vd_loss={l_vd.item()}. Abortando para evitar corromper el modelo "
                    "con gradientes invalidos (M6)."
                )

            if train and device.type == "cuda" and _emit_once.should_emit("cuda_batch_report"):
                cuda_idx = 0 if device.index is None else device.index
                logger.debug(
                    "[CUDA] First training batch on GPU | "
                    "x=%s y=%s pred=%s | allocated=%.1f MiB | reserved=%.1f MiB",
                    x.device, y.device, pred.device,
                    torch.cuda.memory_allocated(cuda_idx) / (1024**2),
                    torch.cuda.memory_reserved(cuda_idx) / (1024**2),
                )
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"[NaN/Inf GUARD] Norma de gradiente no finita en batch "
                        f"{num_batches + 1}: grad_norm={grad_norm.item()}. Abortando para "
                        "evitar corromper el modelo (M6)."
                    )
                scaler.step(optimizer)
                scaler.update()

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


def evaluate_epoch(model, loader, cfg: Config, device: torch.device, calibration: dict,
                   compute_uncertainty: bool = True):
    """sf_r2/vd_r2/sf_rmse/vd_rmse se calculan una sola vez al final sobre el dataset
    completo (acumulación global de SS_res/SS_tot y suma de errores cuadrados). El resto
    de las métricas (loss, incertidumbre) sí son promediables por batch.

    val_loss/R²/RMSE se computan SIEMPRE con un forward determinista (dropout off en
    model.eval()), de modo que la selección de best.pt es consistente entre épocas. La
    incertidumbre MC-Dropout (cara: cfg.uncertainty_passes forwards por batch) solo se
    calcula si compute_uncertainty=True; en caso contrario esas métricas quedan en su
    default trivial (0.0 incertidumbre / 1.0 confianza vía pred_std=0).

    El forward corre bajo torch.autocast si cfg.use_amp esta activo (M2) — misma
    politica de memoria que run_one_epoch, sin necesitar GradScaler (no hay backward)."""
    model.eval()
    autocast_enabled = bool(cfg.use_amp) and device.type == "cuda"
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

            with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=_AMP_DTYPE):
                # Forward determinista (model.eval() => dropout off) para loss/R²/RMSE:
                # base consistente de seleccion de best.pt en todas las épocas.
                pred = model(x, d, inj)
            # Loss/metricas/incertidumbre en float32: bajo AMP pred sale en bfloat16 y
            # torch.quantile (summarize_uncertainty) no acepta ese dtype.
            pred = pred.float()
            loss, metrics = compute_all_metrics(pred, y, cfg)
            if compute_uncertainty:
                with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=_AMP_DTYPE):
                    _, pred_std = predict_with_uncertainty(model, x, d, inj, cfg.uncertainty_passes)
                pred_std = pred_std.float()
            else:
                pred_std = torch.zeros_like(pred)
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
    device = resolve_device(cfg.device, deterministic=cfg.deterministic)
    logger.info("[DEVICE] %s", device)
    seed_everything(cfg.seed)

    train_ds, val_ds, train_path, val_path = build_datasets(cfg)
    train_loader, train_num_workers = build_loader(train_ds, cfg, device, shuffle=True)
    val_loader, val_num_workers = build_loader(val_ds, cfg, device, shuffle=False)

    logger.info(
        "[DATA] train=%d val=%d | workers train=%d val=%d",
        len(train_ds), len(val_ds), train_num_workers, val_num_workers,
    )

    model = PhysicalFNOArchitecture(
        time_steps=cfg.time_steps,
        in_c=5,
        h_dim=cfg.hidden_dim,
        modes=cfg.spectral_modes,
        cond_dim=128,
        dropout_p=cfg.dropout_p,
        use_group_norm=cfg.use_group_norm,
    ).to(device)

    param_groups = build_param_groups(model, weight_decay=cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
    scheduler = build_scheduler(optimizer, cfg)
    amp_enabled = bool(cfg.use_amp) and device.type == "cuda"
    # GradScaler siempre deshabilitado: la ruta AMP usa bfloat16 (_AMP_DTYPE), que no
    # necesita loss scaling; ademas unscale_ no soporta los params ComplexFloat del FNO.
    scaler = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu", enabled=False)
    if amp_enabled:
        logger.info("[AMP] Mixed precision activo en bfloat16 (sin GradScaler; params complejos).")
    if cfg.use_amp and device.type != "cuda":
        logger.warning("[AMP] use_amp=True pero el dispositivo no es CUDA; se ignora (M2 solo aplica en GPU).")

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
    logger.info("[MODEL] Parametros entrenables: %s", f"{count_parameters(model):,}")

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
                candidate, model, optimizer, device, run_signature, scheduler=scheduler
            )
            if resumed:
                resumed_from = resume_path
                history = load_json(history_path, default=[])
                logger.info(
                    "[RESUME] Cargado %s. Continuando desde epoch %d (ultima completa: %d)",
                    resume_path.name, start_epoch, start_epoch - 1,
                )
                if last_metrics is not None:
                    logger.info(
                        "[RESUME] Ultimo val_loss=%s | best val_loss=%.6f",
                        last_metrics.get("val_loss", "n/a"), best_val_loss,
                    )
                break

            logger.warning("[RESUME SKIP] %s no se pudo reanudar:", candidate.name)
            for reason in reasons:
                logger.warning("  - %s", reason)

        if resumed_from is None:
            logger.info("[START] Iniciando una corrida nueva para la configuracion actual")
    else:
        logger.info("[START] Entrenando desde cero")

    if start_epoch > cfg.epochs:
        logger.info("El entrenamiento ya alcanzo la epoch %d.", cfg.epochs)
        return

    # Calibracion inicial perezosa: se carga de disco si existe, si no se arranca con el
    # default trivial. La calibracion real (cara) se computa en las épocas de incertidumbre
    # (uncertainty_eval_interval), no al arrancar — evita pagar uncertainty_passes forwards
    # sobre todo val antes de la primera época.
    calibration = (
        load_json(calibration_path, default=default_uncertainty_calibration())
        if calibration_path.exists()
        else default_uncertainty_calibration()
    )

    early_stopping_counter = 0
    pause_datetime = get_next_pause_datetime(cfg.pause_hour)
    logger.info("[SCHEDULE] Pausa programada a %s", pause_datetime.strftime("%Y-%m-%d %H:%M:%S"))

    for epoch in range(start_epoch, cfg.epochs + 1):
        # train_metrics ya incluye sf_r2/vd_r2/sf_rmse/vd_rmse (acumulados globalmente
        # durante el propio paso de entrenamiento) — evita el forward extra sobre todo
        # train que antes hacia evaluate_epoch(train_loader) cada epoca (A3).
        train_metrics = run_one_epoch(model, train_loader, optimizer, cfg, device, train=True, scaler=scaler)

        # La incertidumbre MC-Dropout (cfg.uncertainty_passes forwards sobre val, muy
        # cara) es un diagnostico PERIODICO: se recalibra y se computa solo cada
        # cfg.uncertainty_eval_interval épocas y en la época final — no cada época.
        # val_loss/R²/RMSE (y la seleccion de best.pt) usan siempre el forward
        # determinista de evaluate_epoch, así que no dependen de esto. Requiere dropout
        # activo (model_has_dropout, ligado a C2); sin dropout la calibracion es trivial.
        do_uncertainty = model_has_dropout(model) and (
            epoch == cfg.epochs
            or (cfg.uncertainty_eval_interval > 0 and epoch % cfg.uncertainty_eval_interval == 0)
        )
        if do_uncertainty:
            calibration = calibrate_uncertainty(model, val_loader, cfg, device)
            save_json(calibration_path, calibration)

        val_metrics = evaluate_epoch(model, val_loader, cfg, device, calibration,
                                     compute_uncertainty=do_uncertainty)

        history_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(history_row)
        save_json(history_path, history)
        save_history_plots(history, out_dir)

        logger.info(
            "E%03d | train_loss=%.5f | val_loss=%.5f | val_sf_rmse=%.5f | "
            "val_vd_rmse=%.5f | val_sf_r2=%.4f | val_vd_r2=%.4f | val_sf_unc=%.4f | "
            "val_vd_unc=%.4f | val_sf_conf=%.4f | val_vd_conf=%.4f",
            epoch, train_metrics["loss"], history_row["val_loss"],
            history_row["val_sf_rmse"], history_row["val_vd_rmse"],
            history_row["val_sf_r2"], history_row["val_vd_r2"],
            history_row["val_sf_uncertainty_mean"], history_row["val_vd_uncertainty_mean"],
            history_row["val_sf_confidence_mean"], history_row["val_vd_confidence_mean"],
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
                scheduler=scheduler,
            )
            logger.info("  nuevo best val_loss=%.6f", best_val_loss)
        else:
            early_stopping_counter += 1
            logger.info(
                "  sin mejora en val_loss (%d/%d)",
                early_stopping_counter, cfg.early_stopping_patience,
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
            scheduler=scheduler,
        )

        # Avanza el scheduler DESPUES de guardar los checkpoints de esta epoca, para
        # que optimizer_state_dict (LR usado en esta epoca) y scheduler_state_dict
        # (punto del ciclo que produjo ese LR) queden guardados en sincronia (M1).
        if scheduler is not None:
            scheduler.step()

        if datetime.now() >= pause_datetime:
            logger.info(
                "[AUTO-PAUSE] Se alcanzo la hora de pausa (%d:00). Checkpoint guardado en %s",
                cfg.pause_hour, latest_ckpt_path.resolve(),
            )
            logger.info("Ejecuta el mismo comando de nuevo y continuara desde la siguiente epoch.")
            return

        if early_stopping_counter >= cfg.early_stopping_patience:
            logger.info(
                "[EARLY STOPPING] Detenido en epoch %d tras %d epochs sin mejorar "
                "val_loss en al menos %s.",
                epoch, cfg.early_stopping_patience, cfg.early_stopping_min_delta,
            )
            logger.info("Best val_loss=%.6f", best_val_loss)
            return

    logger.info("[TRAIN COMPLETE] Entrenamiento finalizado. best_val_loss=%.6f", best_val_loss)
