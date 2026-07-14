import sys
import types

import pytest

from fno_co2.config import Config
from fno_co2.models.fno import PhysicalFNOArchitecture
from fno_co2.models.registry import build_model


def test_build_model_fno_baseline_returns_physical_fno():
    cfg = Config(model_variant="fno_baseline", hidden_dim=16, spectral_modes=4, time_steps=4)
    model = build_model(cfg)
    assert isinstance(model, PhysicalFNOArchitecture)


def test_build_model_unknown_variant_raises_value_error():
    cfg = Config(model_variant="no_existe_esta_variante")
    with pytest.raises(ValueError):
        build_model(cfg)


def test_build_model_variant_without_build_function_raises():
    module_name = "fno_co2.models.variants._test_variant_no_build_fn"
    sys.modules[module_name] = types.ModuleType(module_name)
    try:
        cfg = Config(model_variant="_test_variant_no_build_fn")
        with pytest.raises(ValueError):
            build_model(cfg)
    finally:
        del sys.modules[module_name]


def test_build_model_dispatches_to_variant_module():
    module_name = "fno_co2.models.variants._test_fake_variant"
    fake_module = types.ModuleType(module_name)
    sentinel = object()
    fake_module.build = lambda cfg: sentinel
    sys.modules[module_name] = fake_module
    try:
        cfg = Config(model_variant="_test_fake_variant")
        assert build_model(cfg) is sentinel
    finally:
        del sys.modules[module_name]
