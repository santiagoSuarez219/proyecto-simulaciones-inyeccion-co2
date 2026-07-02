import torch
import torch.nn.functional as F

from modelo_itm.config import Config


def spatial_gradient_loss(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


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
