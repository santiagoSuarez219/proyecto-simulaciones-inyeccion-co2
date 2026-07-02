import pytest
import torch

from modelo_itm.models.fno import PhysicalFNOArchitecture


def test_model_forward_basic():
    model = PhysicalFNOArchitecture(time_steps=4, h_dim=32, modes=8)
    model.eval()

    batch_size, h, w = 2, 32, 32
    x = torch.randn(batch_size, 5, h, w)
    d = torch.randn(batch_size, 1)
    inj = torch.randn(batch_size, 4, 2)

    with torch.no_grad():
        pred = model(x, d, inj)

    assert pred.shape == (batch_size, 4, 2, h, w)
    assert torch.isfinite(pred).all()


def test_model_output_shape():
    model = PhysicalFNOArchitecture(time_steps=8, h_dim=64, modes=16)
    model.eval()

    x = torch.randn(3, 5, 64, 64)
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

    x = torch.randn(2, 5, 32, 32)
    d = torch.randn(2, 1)
    inj = torch.randn(2, 5, 2)

    with torch.no_grad():
        pred = model(x, d, inj)

    assert pred.shape == (2, 10, 2, 32, 32)


def test_model_trainable():
    model = PhysicalFNOArchitecture(time_steps=4, h_dim=32, modes=8)
    model.train()

    x = torch.randn(2, 5, 32, 32, requires_grad=True)
    d = torch.randn(2, 1, requires_grad=True)
    inj = torch.randn(2, 4, 2, requires_grad=True)

    pred = model(x, d, inj)
    loss = pred.sum()
    loss.backward()

    assert model.encoder[0].weight.grad is not None
    assert model.fno_blocks[0].weight.grad is not None
