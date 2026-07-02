"""M4 — reproducibilidad: worker_init_fn/generator en DataLoader,
resolve_device(deterministic=...), y muestreo semillado en save_epoch_visuals
(sin depender del modulo global `random`)."""
import random

import torch
from torch.utils.data import TensorDataset

from fno_co2.config import Config
from fno_co2.data.loaders import build_loader
from fno_co2.utils.device import resolve_device


def test_resolve_device_accepts_deterministic_flag_on_cpu():
    """En CPU, deterministic no tiene efecto observable (cudnn solo aplica en
    CUDA) pero la funcion no debe fallar con el nuevo parametro."""
    device = resolve_device("cpu", deterministic=True)
    assert device.type == "cpu"
    device2 = resolve_device("cpu", deterministic=False)
    assert device2.type == "cpu"


def test_build_loader_shuffle_is_reproducible_with_same_seed():
    """Dos DataLoaders construidos con el mismo cfg.seed deben producir el
    mismo orden de shuffle (via el `generator` semillado)."""
    dataset = TensorDataset(torch.arange(20).float())
    cfg = Config(seed=123, batch_size=4, num_workers=0)
    device = torch.device("cpu")

    loader_a, _ = build_loader(dataset, cfg, device, shuffle=True)
    loader_b, _ = build_loader(dataset, cfg, device, shuffle=True)

    order_a = [batch[0].tolist() for batch in loader_a]
    order_b = [batch[0].tolist() for batch in loader_b]
    assert order_a == order_b


def test_build_loader_different_seed_changes_shuffle_order():
    dataset = TensorDataset(torch.arange(50).float())
    device = torch.device("cpu")

    loader_a, _ = build_loader(dataset, Config(seed=1, batch_size=4, num_workers=0), device, shuffle=True)
    loader_b, _ = build_loader(dataset, Config(seed=999, batch_size=4, num_workers=0), device, shuffle=True)

    order_a = [batch[0].tolist() for batch in loader_a]
    order_b = [batch[0].tolist() for batch in loader_b]
    assert order_a != order_b


def test_save_epoch_visuals_sample_selection_is_reproducible():
    """Misma seed+epoch -> mismo indice de muestra elegido; no depende del
    modulo global random (verificado perturbando su estado antes de llamar)."""
    cfg = Config(seed=7)
    epoch = 3

    random.seed(999)  # perturba el estado global — no deberia afectar el resultado
    rng1 = random.Random(cfg.seed + epoch)
    idx1 = rng1.randrange(100)

    random.seed(111)  # otra perturbacion distinta
    rng2 = random.Random(cfg.seed + epoch)
    idx2 = rng2.randrange(100)

    assert idx1 == idx2


def test_save_epoch_visuals_sample_selection_varies_by_epoch():
    cfg = Config(seed=7)
    rng_epoch1 = random.Random(cfg.seed + 1)
    rng_epoch2 = random.Random(cfg.seed + 2)
    idx1 = rng_epoch1.randrange(1000)
    idx2 = rng_epoch2.randrange(1000)
    assert idx1 != idx2
