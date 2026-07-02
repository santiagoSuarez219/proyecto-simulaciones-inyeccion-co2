import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelo_itm.config import Config
from modelo_itm.training.losses import compute_loss_terms


def torch_r2_score(pred, gt):
    ss_res = torch.sum((gt - pred) ** 2)
    ss_tot = torch.sum((gt - torch.mean(gt)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def compute_rmse(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt) + 1e-12)


def compute_all_metrics(pred, gt, cfg: Config):
    """Loss por batch (promediable). R2/RMSE de SF y VD se acumulan globalmente
    con init/update/finalize_global_regression_metrics — no aditivos por batch."""
    total_loss, l_sf, l_vd = compute_loss_terms(pred, gt, cfg)
    with torch.no_grad():
        metrics = {
            "loss": float(total_loss.item()),
            "sf_loss": float(l_sf.item()),
            "vd_loss": float(l_vd.item()),
        }
    return total_loss, metrics


def init_global_regression_accumulators(keys=("sf", "vd")):
    return {
        key: {"sum_sq_error": 0.0, "sum_gt": 0.0, "sum_gt_sq": 0.0, "count": 0}
        for key in keys
    }


def update_global_regression_accumulators(accumulators, key, pred, gt):
    with torch.no_grad():
        acc = accumulators[key]
        acc["sum_sq_error"] += float(torch.sum((pred - gt) ** 2).item())
        acc["sum_gt"] += float(torch.sum(gt).item())
        acc["sum_gt_sq"] += float(torch.sum(gt ** 2).item())
        acc["count"] += gt.numel()


def finalize_global_regression_metrics(accumulators):
    """R2 = 1 - SS_res/SS_tot y RMSE = sqrt(SS_res/n) calculados sobre el dataset
    completo (SS_tot = sum(gt^2) - n*mean(gt)^2, acumulable sin una segunda pasada)."""
    metrics = {}
    for key, acc in accumulators.items():
        n = acc["count"]
        if n == 0:
            metrics[f"{key}_rmse"] = 0.0
            metrics[f"{key}_r2"] = 0.0
            continue
        sum_sq_error = acc["sum_sq_error"]
        metrics[f"{key}_rmse"] = float(np.sqrt(sum_sq_error / n + 1e-12))
        mean_gt = acc["sum_gt"] / n
        ss_tot = acc["sum_gt_sq"] - n * (mean_gt ** 2)
        metrics[f"{key}_r2"] = float(1.0 - sum_sq_error / (ss_tot + 1e-8))
    return metrics


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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
