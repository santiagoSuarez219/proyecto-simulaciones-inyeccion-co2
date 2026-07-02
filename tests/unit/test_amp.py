"""
M2 — mixed precision (AMP). Estos tests corren en CPU (sin GPU en esta sesion),
por lo que NO ejercitan el path real de float16/GradScaler en CUDA. Solo
verifican que: (a) cfg.use_amp=True no rompe nada en CPU (autocast/scaler se
ignoran automaticamente fuera de CUDA), y (b) el forward sigue siendo correcto.
Verificacion completa en GPU real queda pendiente (requiere hardware CUDA).
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from fno_co2.config import Config
from fno_co2.models.fno import PhysicalFNOArchitecture
from fno_co2.training.loop import run_one_epoch


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
