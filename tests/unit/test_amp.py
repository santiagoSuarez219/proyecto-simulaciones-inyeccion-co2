"""
M2 — mixed precision (AMP). Los tests de CPU verifican que: (a) cfg.use_amp=True no
rompe nada en CPU (autocast/scaler se ignoran fuera de CUDA), y (b) el forward sigue
siendo correcto. El test marcado `slow`+CUDA ejercita el path REAL en GPU y es la
regresion del bug ComplexFloat (M2): la ruta AMP usa bfloat16 sin GradScaler porque
unscale_ no soporta los parametros espectrales complejos del FNO. Verificado en RTX
6000 Ada durante el preflight de entrenamiento.
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fno_co2.config import Config
from fno_co2.inference.uncertainty import calibrate_uncertainty, default_uncertainty_calibration
from fno_co2.models.fno import PhysicalFNOArchitecture
from fno_co2.training.loop import _AMP_DTYPE, evaluate_epoch, run_one_epoch


def _build_dummy_loader(n_samples=4, time_steps=4, h=8, w=8, batch_size=2):
    x = torch.randn(n_samples, 4, h, w)
    d = torch.randn(n_samples, 1)
    inj = torch.randn(n_samples, time_steps, 2)
    y = torch.randn(n_samples, time_steps, 2, h, w)
    return DataLoader(TensorDataset(x, d, inj, y), batch_size=batch_size, shuffle=False)


def test_run_one_epoch_with_use_amp_true_on_cpu_does_not_break():
    """En CPU, use_amp=True debe ser un no-op seguro (M2 solo se activa en CUDA)."""
    time_steps = 4
    cfg = Config(time_steps=time_steps, hidden_dim=16, spectral_modes=4, batch_size=2, use_amp=True)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loader = _build_dummy_loader(time_steps=time_steps, batch_size=2)

    result = run_one_epoch(model, loader, optimizer, cfg, torch.device("cpu"), train=True)

    for key in ("loss", "sf_loss", "vd_loss", "sf_r2", "vd_r2", "sf_rmse", "vd_rmse"):
        assert torch.isfinite(torch.tensor(result[key])), f"'{key}' no es finito con use_amp=True en CPU"


def test_default_grad_scaler_is_noop_when_disabled():
    """Un GradScaler con enabled=False debe comportarse como si no existiera."""
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    x = torch.tensor(2.0, requires_grad=True)
    loss = x * 3.0
    scaled = scaler.scale(loss)
    assert scaled.item() == loss.item()  # sin escalado real


def test_amp_dtype_is_bfloat16():
    """La ruta AMP debe usar bfloat16 (no float16) para no depender de un GradScaler
    incompatible con los params ComplexFloat del FNO (regresion del bug M2)."""
    assert _AMP_DTYPE is torch.bfloat16


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere GPU CUDA (path real de AMP)")
def test_run_one_epoch_with_amp_on_cuda_handles_complex_params():
    """Regresion M2: en CUDA con use_amp=True, run_one_epoch NO debe crashear por los
    gradientes ComplexFloat de FiLMSpectralBlock (el bug era GradScaler.unscale_ sobre
    params complejos). Con bfloat16 sin GradScaler debe completar y dar loss finita."""
    device = torch.device("cuda")
    time_steps = 4
    cfg = Config(time_steps=time_steps, hidden_dim=16, spectral_modes=4, batch_size=2, use_amp=True)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loader = _build_dummy_loader(time_steps=time_steps, batch_size=2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    result = run_one_epoch(model, loader, optimizer, cfg, device, train=True, scaler=scaler)

    for key in ("loss", "sf_loss", "vd_loss", "sf_r2", "vd_r2", "sf_rmse", "vd_rmse"):
        assert torch.isfinite(torch.tensor(result[key])), f"'{key}' no finito con AMP en CUDA"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere GPU CUDA (path real de AMP)")
@pytest.mark.parametrize("compute_uncertainty", [False, True])
def test_evaluate_epoch_amp_on_cuda_handles_bf16(compute_uncertainty):
    """Regresion: bajo AMP el forward sale en bfloat16; summarize_uncertainty usa
    torch.quantile, que NO acepta bf16. evaluate_epoch debe castear a float32 y no
    crashear, con o sin incertidumbre."""
    device = torch.device("cuda")
    time_steps = 4
    cfg = Config(time_steps=time_steps, hidden_dim=16, spectral_modes=4, batch_size=2,
                 use_amp=True, dropout_p=0.2, uncertainty_passes=3)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4, dropout_p=0.2).to(device)
    loader = _build_dummy_loader(time_steps=time_steps, batch_size=2)

    result = evaluate_epoch(model, loader, cfg, device, default_uncertainty_calibration(),
                            compute_uncertainty=compute_uncertainty)

    for key in ("loss", "sf_r2", "vd_r2", "sf_rmse", "vd_rmse",
                "sf_uncertainty_mean", "sf_confidence_mean"):
        assert torch.isfinite(torch.tensor(result[key])), f"'{key}' no finito bajo AMP"


def test_calibrate_uncertainty_with_use_amp_true_on_cpu_does_not_break():
    """En CPU, use_amp=True debe ser un no-op seguro para calibrate_uncertainty tambien
    (regresion: antes de este fix corria siempre en fp32 sin mirar cfg.use_amp en
    absoluto — medido en GPU real, spec-004 §7.0: ~107 min con o sin AMP, cero mejora)."""
    time_steps = 4
    cfg = Config(time_steps=time_steps, use_amp=True, uncertainty_passes=5)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4, dropout_p=0.2)
    loader = _build_dummy_loader(time_steps=time_steps, batch_size=2)

    calibration = calibrate_uncertainty(model, loader, cfg, torch.device("cpu"))

    for key in ("sf", "vd"):
        assert torch.isfinite(torch.tensor(calibration[key]["alpha"]))
        assert torch.isfinite(torch.tensor(calibration[key]["error_q95"]))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere GPU CUDA (path real de AMP)")
def test_calibrate_uncertainty_amp_on_cuda_handles_bf16():
    """Regresion: bajo AMP predict_with_uncertainty sale en bfloat16; torch.quantile
    (via _quantile_capped) no lo acepta. calibrate_uncertainty debe castear a float32
    antes de acumular y no crashear con use_amp=True en CUDA."""
    device = torch.device("cuda")
    time_steps = 4
    cfg = Config(time_steps=time_steps, hidden_dim=16, spectral_modes=4, batch_size=2,
                 use_amp=True, dropout_p=0.2, uncertainty_passes=3)
    model = PhysicalFNOArchitecture(time_steps=time_steps, h_dim=16, modes=4, dropout_p=0.2).to(device)
    loader = _build_dummy_loader(time_steps=time_steps, batch_size=2)

    calibration = calibrate_uncertainty(model, loader, cfg, device)

    for key in ("sf", "vd"):
        assert torch.isfinite(torch.tensor(calibration[key]["alpha"]))
        assert torch.isfinite(torch.tensor(calibration[key]["error_q95"]))


def test_film_spectral_block_fft_stays_finite_under_simulated_autocast():
    """M2: la FFT dentro de FiLMSpectralBlock debe seguir siendo numericamente
    estable (forzada a float32) incluso si el bloque se invoca dentro de un
    contexto autocast activo (simulado aqui con bfloat16 en CPU)."""
    from fno_co2.models.blocks import FiLMSpectralBlock

    block = FiLMSpectralBlock(c=8, modes=4, cond_dim=16)
    x = torch.randn(2, 8, 10, 10)
    cond = torch.randn(2, 16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        out = block(x, cond)

    assert torch.isfinite(out).all()
    assert out.shape == (2, 8, 10, 10)
