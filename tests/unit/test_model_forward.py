import pytest
import torch

from modelo_itm.models.fno import PhysicalFNOArchitecture

# in_c=5 (default) es el total de canales que entran al encoder DESPUES de
# concatenar profundidad (forward: torch.cat([x, depth_map], dim=1)). Con el
# dataset real (4 propiedades estaticas: AFI/COH/PERM/PORO), x trae 4 canales
# y depth_map 1 -> 4+1=5. Estos tests usan x con 4 canales para reflejar eso.


def test_model_forward_basic():
    model = PhysicalFNOArchitecture(time_steps=4, h_dim=32, modes=8)
    model.eval()

    batch_size, h, w = 2, 32, 32
    x = torch.randn(batch_size, 4, h, w)
    d = torch.randn(batch_size, 1)
    inj = torch.randn(batch_size, 4, 2)

    with torch.no_grad():
        pred = model(x, d, inj)

    assert pred.shape == (batch_size, 4, 2, h, w)
    assert torch.isfinite(pred).all()


def test_model_output_shape():
    model = PhysicalFNOArchitecture(time_steps=8, h_dim=64, modes=16)
    model.eval()

    x = torch.randn(3, 4, 64, 64)
    d = torch.randn(3, 1)
    inj = torch.randn(3, 8, 2)

    with torch.no_grad():
        pred = model(x, d, inj)

    assert pred.shape[0] == 3
    assert pred.shape[1] == 8
    assert pred.shape[2] == 2
    assert pred.shape[3] == 64
    assert pred.shape[4] == 64


def test_model_injection_padding():
    model = PhysicalFNOArchitecture(time_steps=10, h_dim=32, modes=8)
    model.eval()

    x = torch.randn(2, 4, 32, 32)
    d = torch.randn(2, 1)
    inj = torch.randn(2, 5, 2)

    with torch.no_grad():
        pred = model(x, d, inj)

    assert pred.shape == (2, 10, 2, 32, 32)


def test_model_trainable():
    model = PhysicalFNOArchitecture(time_steps=4, h_dim=32, modes=8)
    model.train()

    x = torch.randn(2, 4, 32, 32, requires_grad=True)
    d = torch.randn(2, 1, requires_grad=True)
    inj = torch.randn(2, 4, 2, requires_grad=True)

    pred = model(x, d, inj)
    loss = pred.sum()
    loss.backward()

    assert model.encoder[0].weight.grad is not None
    assert model.fno_blocks[0].weight.grad is not None


def test_model_in_c_matches_concatenated_depth_channel():
    """Regresion: in_c debe ser el TOTAL tras concatenar profundidad, no los
    canales de x antes de concatenar (bug introducido en la migracion de Fase 2
    y corregido en Fase 9 — el encoder original NO sumaba +1 a in_c)."""
    in_c = 5
    model = PhysicalFNOArchitecture(time_steps=4, h_dim=16, modes=4, in_c=in_c)
    assert model.encoder[0].in_channels == in_c

    x = torch.randn(2, in_c - 1, 16, 16)  # 4 propiedades estaticas
    d = torch.randn(2, 1)
    inj = torch.randn(2, 4, 2)
    with torch.no_grad():
        pred = model(x, d, inj)
    assert torch.isfinite(pred).all()
