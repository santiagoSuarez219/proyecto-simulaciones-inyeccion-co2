import pytest
import torch

from modelo_itm.data.dataset import DatasetLayers, load_injection_series


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


def _write_injection_pt(path, values):
    torch.save(
        {
            "tensor": torch.tensor(values, dtype=torch.float32),
            "name": path.stem,
            "normalization": {"applied": True, "method": "minmax"},
        },
        path,
    )


def test_load_injection_series_no_paths_returns_zeros():
    inj = load_injection_series([], time_steps=10)
    assert inj.shape == (10, 2)
    assert torch.all(inj == 0.0)


def test_load_injection_series_passes_through_etl_normalized_values(tmp_path):
    """A2: el dataset ya NO re-normaliza (sin log1p/reescalado local) — los
    valores ya normalizados por el ETL en [0,1] deben llegar intactos."""
    p1 = tmp_path / "injection_tene_1.pt"
    p2 = tmp_path / "injection_tene_2.pt"
    values1 = [0.0, 0.25, 0.5, 0.75, 1.0]
    values2 = [1.0, 0.8, 0.6, 0.4, 0.2]
    _write_injection_pt(p1, values1)
    _write_injection_pt(p2, values2)

    inj = load_injection_series([p1, p2], time_steps=5)

    assert inj.shape == (5, 2)
    assert torch.allclose(inj[:, 0], torch.tensor(values1))
    assert torch.allclose(inj[:, 1], torch.tensor(values2))


def test_load_injection_series_pads_and_truncates(tmp_path):
    p1 = tmp_path / "injection_tene_1.pt"
    p2 = tmp_path / "injection_tene_2.pt"
    _write_injection_pt(p1, [0.1, 0.2])  # shorter than time_steps -> padded with 0
    _write_injection_pt(p2, [0.9, 0.8, 0.7, 0.6, 0.5])  # longer -> truncated

    inj = load_injection_series([p1, p2], time_steps=4)

    assert inj.shape == (4, 2)
    assert torch.allclose(inj[:, 0], torch.tensor([0.1, 0.2, 0.0, 0.0]))
    assert torch.allclose(inj[:, 1], torch.tensor([0.9, 0.8, 0.7, 0.6]))


def test_load_injection_series_sanitizes_nan_inf(tmp_path):
    p1 = tmp_path / "injection_tene_1.pt"
    p2 = tmp_path / "injection_tene_2.pt"
    _write_injection_pt(p1, [float("nan"), float("inf"), float("-inf"), 0.5])
    _write_injection_pt(p2, [0.1, 0.2, 0.3, 0.4])

    inj = load_injection_series([p1, p2], time_steps=4)

    assert torch.isfinite(inj).all()
    assert torch.allclose(inj[:, 0], torch.tensor([0.0, 0.0, 0.0, 0.5]))
