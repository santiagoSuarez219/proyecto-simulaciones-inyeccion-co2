import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def train_script():
    """scripts/train.py no es un paquete (sin __init__.py); se carga por ruta de archivo
    para poder testear build_parser()/resolve_config() sin ejecutar el bloque __main__."""
    return _load_script_module("_train_script_under_test", "train.py")


@pytest.fixture(scope="session")
def aggregate_script():
    return _load_script_module("_aggregate_script_under_test", "aggregate_experiments.py")
