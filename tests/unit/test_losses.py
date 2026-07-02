import pytest
import torch

from fno_co2.config import Config
from fno_co2.training.losses import _segment_boundaries, compute_loss_terms, spatial_gradient_loss


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


def test_segment_boundaries_preserves_original_61_steps_default():
    """time_steps=61 debe reproducir exactamente los limites originales
    hardcodeados (1, 21) -> segmentos [0:1], [1:21], [21:61]."""
    assert _segment_boundaries(61) == (1, 21)


@pytest.mark.parametrize("time_steps", [3, 4, 10, 30, 61, 97, 200])
def test_segment_boundaries_cover_exactly_without_overlap(time_steps):
    b1, b2 = _segment_boundaries(time_steps)
    assert 0 < b1 < b2 < time_steps
    # Los 3 segmentos [0,b1) + [b1,b2) + [b2,time_steps) cubren [0, time_steps)
    # exactamente una vez cada indice, sin huecos ni solapamientos.
    segments = [set(range(0, b1)), set(range(b1, b2)), set(range(b2, time_steps))]
    covered = segments[0] | segments[1] | segments[2]
    assert covered == set(range(time_steps))
    assert len(segments[0]) + len(segments[1]) + len(segments[2]) == time_steps


def test_segment_boundaries_rejects_too_few_timesteps():
    with pytest.raises(ValueError):
        _segment_boundaries(2)


@pytest.mark.parametrize("time_steps", [4, 30, 97])
def test_compute_loss_terms_works_for_non_default_time_steps(time_steps):
    """A1: la loss debe ser correcta (finita, sin errores de shape) para
    valores de time_steps distintos del default hardcodeado de 61."""
    cfg = Config(time_steps=time_steps)
    pred = torch.randn(2, time_steps, 2, 8, 8)
    gt = torch.randn(2, time_steps, 2, 8, 8)

    total_loss, l_sf, l_vd = compute_loss_terms(pred, gt, cfg)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(l_sf)
    assert torch.isfinite(l_vd)
    assert total_loss.item() > 0


def test_compute_loss_terms_time_steps_derived_from_tensor_not_config():
    """La loss usa pred.shape[1] (el tensor real), no cfg.time_steps — evita
    desincronizacion si cfg queda desactualizada respecto al tensor real."""
    cfg = Config(time_steps=999)  # deliberadamente distinto del tensor
    pred = torch.randn(2, 30, 2, 8, 8)
    gt = torch.randn(2, 30, 2, 8, 8)

    total_loss, l_sf, l_vd = compute_loss_terms(pred, gt, cfg)

    assert torch.isfinite(total_loss)
