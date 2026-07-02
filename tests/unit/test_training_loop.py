import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fno_co2.config import Config
from fno_co2.models.fno import PhysicalFNOArchitecture
from fno_co2.training.loop import run_one_epoch


def _build_dummy_loader(n_samples=6, time_steps=4, h=8, w=8, batch_size=2):
    x = torch.randn(n_samples, 4, h, w)
    d = torch.randn(n_samples, 1)
    inj = torch.randn(n_samples, time_steps, 2)
    y = torch.randn(n_samples, time_steps, 2, h, w)
    dataset = TensorDataset(x, d, inj, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_run_one_epoch_returns_global_regression_metrics():
    """A3: run_one_epoch acumula sf_r2/vd_r2/sf_rmse/vd_rmse durante el propio
    paso de entrenamiento (sin necesitar un evaluate_epoch(train_loader) extra)."""
    time_steps = 4
    cfg = Config(time_steps=time_steps, hidden_dim=16, spectral_modes=4, batch_size=2)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loader = _build_dummy_loader(time_steps=time_steps, batch_size=2)

    result = run_one_epoch(model, loader, optimizer, cfg, torch.device("cpu"), train=True)

    for key in ("loss", "sf_loss", "vd_loss", "sf_r2", "vd_r2", "sf_rmse", "vd_rmse"):
        assert key in result, f"falta '{key}' en el resultado de run_one_epoch"
        assert torch.isfinite(torch.tensor(result[key])), f"'{key}' no es finito: {result[key]}"

    assert result["sf_rmse"] >= 0.0
    assert result["vd_rmse"] >= 0.0


def test_run_one_epoch_regression_metrics_match_manual_accumulation():
    """Los valores devueltos deben coincidir con acumular manualmente sobre
    los mismos batches (sin backprop, comparando forward puro)."""
    from fno_co2.training.metrics import (
        finalize_global_regression_metrics,
        init_global_regression_accumulators,
        update_global_regression_accumulators,
    )

    time_steps = 4
    cfg = Config(time_steps=time_steps, hidden_dim=16, spectral_modes=4, batch_size=2)
    torch.manual_seed(123)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4)
    model.eval()
    loader = _build_dummy_loader(time_steps=time_steps, batch_size=2)

    acc = init_global_regression_accumulators(("sf", "vd"))
    with torch.no_grad():
        for x, d, inj, y in loader:
            pred = model(x, d, inj)
            update_global_regression_accumulators(acc, "sf", pred[:, :, 0], y[:, :, 0])
            update_global_regression_accumulators(acc, "vd", pred[:, :, 1], y[:, :, 1])
    expected = finalize_global_regression_metrics(acc)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)  # lr=0 -> no cambia pesos
    result = run_one_epoch(model, loader, optimizer, cfg, torch.device("cpu"), train=False)

    assert result["sf_r2"] == expected["sf_r2"]
    assert result["vd_r2"] == expected["vd_r2"]
    assert result["sf_rmse"] == expected["sf_rmse"]
    assert result["vd_rmse"] == expected["vd_rmse"]


def test_run_one_epoch_aborts_on_nan_loss():
    """M6: una loss no finita (ground truth con NaN) debe abortar el
    entrenamiento con un error claro, no propagarse silenciosamente."""
    time_steps = 4
    cfg = Config(time_steps=time_steps, hidden_dim=16, spectral_modes=4, batch_size=2)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    n_samples, h, w = 2, 8, 8
    x = torch.randn(n_samples, 4, h, w)
    d = torch.randn(n_samples, 1)
    inj = torch.randn(n_samples, time_steps, 2)
    y = torch.full((n_samples, time_steps, 2, h, w), float("nan"))
    loader = DataLoader(TensorDataset(x, d, inj, y), batch_size=2)

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        run_one_epoch(model, loader, optimizer, cfg, torch.device("cpu"), train=True)


def test_run_one_epoch_aborts_on_inf_loss():
    time_steps = 4
    cfg = Config(time_steps=time_steps, hidden_dim=16, spectral_modes=4, batch_size=2)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    n_samples, h, w = 2, 8, 8
    x = torch.randn(n_samples, 4, h, w)
    d = torch.randn(n_samples, 1)
    inj = torch.randn(n_samples, time_steps, 2)
    y = torch.full((n_samples, time_steps, 2, h, w), float("inf"))
    loader = DataLoader(TensorDataset(x, d, inj, y), batch_size=2)

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        run_one_epoch(model, loader, optimizer, cfg, torch.device("cpu"), train=True)


def test_run_one_epoch_does_not_raise_on_finite_data():
    """Red de seguridad: confirma que el guard no genera falsos positivos con
    datos normales (ya cubierto por otros tests, pero explicito aqui)."""
    time_steps = 4
    cfg = Config(time_steps=time_steps, hidden_dim=16, spectral_modes=4, batch_size=2)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loader = _build_dummy_loader(time_steps=time_steps, batch_size=2)

    result = run_one_epoch(model, loader, optimizer, cfg, torch.device("cpu"), train=True)
    assert torch.isfinite(torch.tensor(result["loss"]))
