import torch
import torch.nn.functional as F

from modelo_itm.config import Config


def spatial_gradient_loss(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


def _segment_boundaries(time_steps: int) -> tuple[int, int]:
    """Deriva los limites (b1, b2) de los 3 segmentos temporales de la loss de SF
    (t0=[0,b1) / t_mid=[b1,b2) / t_tail=[b2,time_steps)), preservando las proporciones
    del diseno original (limites 1/21 sobre 61 timesteps totales) en vez de asumir
    time_steps=61 fijo. Con time_steps=61 reproduce exactamente (1, 21)."""
    if time_steps < 3:
        raise ValueError(
            f"time_steps={time_steps} es insuficiente para los 3 segmentos de la loss "
            "de SF (t0 / t_mid / t_tail); se requieren al menos 3 timesteps."
        )
    b1 = max(1, round(time_steps * 1 / 61))
    b2 = round(time_steps * 21 / 61)
    b2 = max(b2, b1 + 1)
    b2 = min(b2, time_steps - 1)
    return b1, b2


def compute_loss_terms(pred, gt, cfg: Config):
    def seg_l(p, g):
        l1 = F.smooth_l1_loss(p, g)
        grad = spatial_gradient_loss(p, g)
        return l1 + cfg.grad_weight * grad

    time_steps = pred.shape[1]
    b1, b2 = _segment_boundaries(time_steps)
    assert 0 < b1 < b2 < time_steps, (
        f"segmentos de loss invalidos para time_steps={time_steps}: b1={b1}, b2={b2}"
    )

    l_sf = (
        cfg.seg_t0_weight * seg_l(pred[:, 0:b1, 0], gt[:, 0:b1, 0])
        + cfg.seg_t1_20_weight * seg_l(pred[:, b1:b2, 0], gt[:, b1:b2, 0])
        + cfg.seg_t21_60_weight * seg_l(pred[:, b2:time_steps, 0], gt[:, b2:time_steps, 0])
    )
    l_vd = seg_l(pred[:, :, 1], gt[:, :, 1])
    total_loss = cfg.sf_weight * l_sf + cfg.vd_weight * l_vd
    return total_loss, l_sf, l_vd
