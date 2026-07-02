import torch
from torch.utils.data import DataLoader, TensorDataset

from fno_co2.config import Config
from fno_co2.inference.uncertainty import (
    calibrate_uncertainty,
    default_uncertainty_calibration,
    model_has_dropout,
    predict_with_uncertainty,
)
from fno_co2.models.fno import PhysicalFNOArchitecture


def test_model_has_dropout_true_with_default_config():
    """C2: PhysicalFNOArchitecture() con el dropout_p default (0.1, no 0.0) debe
    tener capas Dropout2d reales — la feature de incertidumbre ya no es codigo
    muerto."""
    model = PhysicalFNOArchitecture(time_steps=4, h_dim=16, modes=4)
    assert model_has_dropout(model) is True


def test_model_has_dropout_false_when_p_zero_is_still_true_but_std_is_zero():
    """Con dropout_p=0.0 la capa Dropout2d sigue presente (model_has_dropout=True)
    pero es un no-op: MC Dropout no debe producir varianza real. Caso limite
    documentado, no relevante con el default (0.1)."""
    model = PhysicalFNOArchitecture(time_steps=4, h_dim=16, modes=4, dropout_p=0.0)
    assert model_has_dropout(model) is True  # la capa existe, aunque p=0.0

    x = torch.randn(2, 4, 8, 8)
    d = torch.randn(2, 1)
    inj = torch.randn(2, 4, 2)
    _, pred_std = predict_with_uncertainty(model, x, d, inj, passes=10)
    assert torch.allclose(pred_std, torch.zeros_like(pred_std), atol=1e-6)


def test_mc_dropout_produces_nontrivial_std():
    """Criterio de aceptacion de C2: MC Dropout produce desviacion estandar
    NO trivial (antes: siempre 0.0 porque el modelo no tenia Dropout)."""
    torch.manual_seed(0)
    model = PhysicalFNOArchitecture(time_steps=4, h_dim=16, modes=4, dropout_p=0.2)

    x = torch.randn(2, 4, 8, 8)
    d = torch.randn(2, 1)
    inj = torch.randn(2, 4, 2)

    pred_mean, pred_std = predict_with_uncertainty(model, x, d, inj, passes=30)

    assert pred_mean.shape == pred_std.shape == (2, 4, 2, 8, 8)
    assert torch.isfinite(pred_std).all()
    assert pred_std.max().item() > 1e-4, "MC Dropout sigue siendo codigo muerto (std ~ 0)"


def test_predict_with_uncertainty_single_pass_returns_zero_std():
    """passes<=1 debe seguir devolviendo std=0 (sin overhead de MC Dropout)."""
    model = PhysicalFNOArchitecture(time_steps=4, h_dim=16, modes=4, dropout_p=0.2)
    x = torch.randn(1, 4, 8, 8)
    d = torch.randn(1, 1)
    inj = torch.randn(1, 4, 2)

    _, pred_std = predict_with_uncertainty(model, x, d, inj, passes=1)
    assert torch.all(pred_std == 0.0)


def test_calibrate_uncertainty_produces_nondefault_alpha_with_dropout():
    """Con dropout activo, calibrate_uncertainty ya no debe hacer short-circuit
    al default trivial — debe iterar el val_loader y calcular alpha/error_q95
    reales a partir de la relacion error/std observada."""
    torch.manual_seed(0)
    time_steps = 4
    cfg = Config(time_steps=time_steps, uncertainty_passes=10)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4, dropout_p=0.2)

    n = 4
    x = torch.randn(n, 4, 8, 8)
    d = torch.randn(n, 1)
    inj = torch.randn(n, time_steps, 2)
    y = torch.randn(n, time_steps, 2, 8, 8)
    loader = DataLoader(TensorDataset(x, d, inj, y), batch_size=2)

    calibration = calibrate_uncertainty(model, loader, cfg, torch.device("cpu"))

    default = default_uncertainty_calibration()
    assert calibration != default, "calibracion sigue siendo el default trivial (0.0/1.0) con dropout activo"
    for key in ("sf", "vd"):
        assert calibration[key]["error_q95"] > 0.0
