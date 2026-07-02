import pytest
import torch

from modelo_itm.data.dataset import DatasetLayers


@pytest.fixture
def dummy_dataset(tmp_path):
    case_dir = tmp_path / "case_001"
    case_dir.mkdir()

    target_dir = case_dir / "layer_cubes"
    target_dir.mkdir()

    static_dirs = {
        "afi_layer_cubes": case_dir / "afi_layer_cubes",
        "cohesion_layer_cubes": case_dir / "cohesion_layer_cubes",
        "permeability_layer_cubes": case_dir / "permeability_layer_cubes",
        "porosity_layer_cubes": case_dir / "porosity_layer_cubes",
    }

    for dir_name, dir_path in static_dirs.items():
        dir_path.mkdir()

    for k in range(1, 3):
        y = torch.randn(2, 61, 100, 100)
        torch.save(y, target_dir / f"{k}.pt")

        for key, dir_path in static_dirs.items():
            x = torch.randn(100, 100)
            torch.save(x, dir_path / f"{k}.pt")

    return tmp_path


def test_dataset_layers_init(dummy_dataset):
    ds = DatasetLayers(dummy_dataset, max_layer=60)
    assert len(ds) > 0


def test_dataset_layers_getitem(dummy_dataset):
    ds = DatasetLayers(dummy_dataset, max_layer=60)
    if len(ds) > 0:
        x, depth, inj, y = ds[0]
        assert x.shape[0] == 4
        assert depth.shape == (1,)
        assert inj.shape == (y.shape[0], 2)
        assert y.shape[1] == 2


def test_dataset_layers_shapes(dummy_dataset):
    ds = DatasetLayers(dummy_dataset, max_layer=60)
    if len(ds) > 0:
        for i in range(min(len(ds), 2)):
            x, depth, inj, y = ds[i]
            assert x.dtype == torch.float32
            assert depth.dtype == torch.float32
            assert inj.dtype == torch.float32
            assert y.dtype == torch.float32
