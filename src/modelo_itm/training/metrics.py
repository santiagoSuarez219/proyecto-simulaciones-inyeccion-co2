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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
