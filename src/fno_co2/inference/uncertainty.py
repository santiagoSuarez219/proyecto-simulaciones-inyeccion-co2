import torch
import torch.nn as nn
from tqdm import tqdm

from fno_co2.config import Config, EPS
from fno_co2.utils import EmitOnce, get_logger

logger = get_logger(__name__)

_emit_once = EmitOnce()


def model_has_dropout(model) -> bool:
    return any(isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) for module in model.modules())


def _quantile_capped(values: torch.Tensor, q: float, max_elems: int = 2**24 - 1) -> float:
    """torch.quantile falla con "input tensor is too large" si el tensor tiene mas de
    ~2^24 elementos. En calibracion el error se acumula sobre TODO el val set (cientos de
    millones de valores), asi que para tensores grandes se estima el cuantil sobre una
    submuestra aleatoria deterministica (con reemplazo, memoria acotada) — precision de
    sobra para un q95. Tensores pequenos usan torch.quantile directo."""
    flat = values.reshape(-1)
    n = flat.numel()
    if n > max_elems:
        g = torch.Generator(device=flat.device).manual_seed(0)
        idx = torch.randint(0, n, (max_elems,), generator=g, device=flat.device)
        flat = flat[idx]
    return float(torch.quantile(flat, q).item())


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

    if not dropout_modules and _emit_once.should_emit("mc_dropout_warning"):
        logger.warning(
            "[UNCERTAINTY WARNING] El modelo no tiene capas Dropout activas; "
            "MC Dropout devolvera desviacion estandar cero."
        )

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
    # Antes de esta corrección, calibrate_uncertainty corría siempre en fp32 completo sin
    # importar cfg.use_amp (a diferencia de la rama de incertidumbre en evaluate_epoch,
    # loop.py, que ya usaba autocast) — medido: ~107 min con o sin AMP, cero mejora. Mismo
    # patron de autocast que el resto del pipeline (loop.py::_AMP_DTYPE = bfloat16).
    autocast_enabled = bool(cfg.use_amp) and device.type == "cuda"
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

            with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=torch.bfloat16):
                pred_mean, pred_std = predict_with_uncertainty(model, x, d, inj, cfg.uncertainty_passes)
            # bajo AMP los preds salen en bfloat16; torch.quantile (_quantile_capped) no lo acepta.
            pred_mean = pred_mean.float()
            pred_std = pred_std.float()
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
    sf_error_q95 = _quantile_capped(sf_abs_error_tensor, 0.95)
    vd_error_q95 = _quantile_capped(vd_abs_error_tensor, 0.95)

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
        "sf_uncertainty_p95": _quantile_capped(sf_unc_flat, 0.95),
        "vd_uncertainty_p95": _quantile_capped(vd_unc_flat, 0.95),
    }


def load_or_create_uncertainty_calibration(path, model, val_loader, cfg: Config, device: torch.device):
    from fno_co2.utils.io import load_json, save_json

    if not model_has_dropout(model):
        calibration = default_uncertainty_calibration()
        save_json(path, calibration)
        logger.info("[UNCERTAINTY] Modelo sin Dropout: calibracion desactivada para no frenar el entrenamiento.")
        return calibration

    if path.exists():
        return load_json(path, default={})

    calibration = calibrate_uncertainty(model, val_loader, cfg, device)
    save_json(path, calibration)
    return calibration
