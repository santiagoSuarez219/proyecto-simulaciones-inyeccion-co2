import pytest
import torch

from modelo_itm.config import Config
from modelo_itm.training.losses import compute_loss_terms, spatial_gradient_loss


def test_spatial_gradient_loss():
    pred = torch.randn(2, 4, 10, 10)
    gt = torch.randn(2, 4, 10, 10)
    loss = spatial_gradient_loss(pred, gt)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_compute_loss_terms_default_config():
    cfg = Config(time_steps=61)
    pred = torch.randn(2, 61, 2, 10, 10)
    gt = torch.randn(2, 61, 2, 10, 10)

    total_loss, l_sf, l_vd = compute_loss_terms(pred, gt, cfg)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(l_sf)
    assert torch.isfinite(l_vd)
    assert total_loss.item() > 0


def test_compute_loss_terms_output_scalar():
    cfg = Config(time_steps=61)
    pred = torch.randn(4, 61, 2, 32, 32)
    gt = torch.randn(4, 61, 2, 32, 32)

    total_loss, l_sf, l_vd = compute_loss_terms(pred, gt, cfg)

    assert total_loss.dim() == 0
    assert l_sf.dim() == 0
    assert l_vd.dim() == 0


def test_compute_loss_terms_with_custom_weights():
    cfg = Config(
        time_steps=61,
        sf_weight=1.0,
        vd_weight=2.0,
        grad_weight=0.5,
    )
    pred = torch.randn(2, 61, 2, 16, 16)
    gt = torch.randn(2, 61, 2, 16, 16)

    total_loss, l_sf, l_vd = compute_loss_terms(pred, gt, cfg)

    assert torch.isfinite(total_loss)
