import pytest
import torch

from fno_co2.data.dataset import DatasetLayers, load_injection_series


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


@pytest.fixture
def multi_case_dataset(tmp_path):
    """N casos independientes, cada uno con su propia serie de inyeccion —
    para ejercitar el cache LRU de DatasetLayers (B4)."""
    n_cases = 6
    static_names = ["afi_layer_cubes", "cohesion_layer_cubes", "permeability_layer_cubes", "porosity_layer_cubes"]

    for case_idx in range(n_cases):
        case_dir = tmp_path / f"case_{case_idx:03d}"
        target_dir = case_dir / "layer_cubes"
        target_dir.mkdir(parents=True)
        inj_dir = case_dir / "injection_name_tensors"
        inj_dir.mkdir(parents=True)

        for name in static_names:
            (case_dir / name).mkdir(parents=True)

        torch.save(torch.randn(2, 4, 6, 6), target_dir / "1.pt")
        for name in static_names:
            torch.save(torch.randn(6, 6), case_dir / name / "1.pt")
        for well in ("tene_1", "tene_2"):
            _write_injection_pt(inj_dir / f"injection_{well}.pt", [0.1, 0.2, 0.3, 0.4])

    return tmp_path, n_cases


def test_dataset_layers_inj_cache_respects_max_size(multi_case_dataset):
    """B4: el cache de inyeccion no debe crecer sin limite — cada caso tiene
    su propia serie, asi que acceder a mas casos que max_inj_cache_size debe
    evictar los mas antiguos (LRU)."""
    root, n_cases = multi_case_dataset
    max_cache_size = 3
    ds = DatasetLayers(root, max_layer=3, max_inj_cache_size=max_cache_size)
    assert len(ds) == n_cases  # 1 capa por caso en este fixture

    for i in range(len(ds)):
        ds[i]

    assert len(ds._inj_cache) <= max_cache_size


def test_dataset_layers_inj_cache_default_is_unbounded_but_reasonable(multi_case_dataset):
    root, n_cases = multi_case_dataset
    ds = DatasetLayers(root, max_layer=3)  # default max_inj_cache_size=512
    for i in range(len(ds)):
        ds[i]
    assert len(ds._inj_cache) == n_cases  # todos caben bajo el default


@pytest.fixture
def dataset_with_mismatched_injection_length(tmp_path):
    """M8: DatasetLayers con una serie de inyeccion mas CORTA que target
    (y.shape[0]) — debe forzar el padding con ceros dentro de __getitem__."""
    case_dir = tmp_path / "case_001"
    target_dir = case_dir / "layer_cubes"
    target_dir.mkdir(parents=True)
    inj_dir = case_dir / "injection_name_tensors"
    inj_dir.mkdir(parents=True)

    static_names = ["afi_layer_cubes", "cohesion_layer_cubes", "permeability_layer_cubes", "porosity_layer_cubes"]
    for name in static_names:
        (case_dir / name).mkdir(parents=True)

    time_steps_target = 10  # y tendra 10 timesteps
    injection_length = 4  # la serie de inyeccion es mas corta -> requiere padding

    torch.save(torch.randn(2, time_steps_target, 5, 5), target_dir / "1.pt")
    for name in static_names:
        torch.save(torch.randn(5, 5), case_dir / name / "1.pt")

    for well in ("tene_1", "tene_2"):
        _write_injection_pt(inj_dir / f"injection_{well}.pt", [0.1, 0.2, 0.3, 0.4][:injection_length])

    return tmp_path, time_steps_target, injection_length


def test_dataset_layers_pads_short_injection_series(dataset_with_mismatched_injection_length):
    root, time_steps_target, injection_length = dataset_with_mismatched_injection_length
    ds = DatasetLayers(root, max_layer=time_steps_target - 1)
    assert len(ds) == 1

    x, depth, inj, y = ds[0]

    assert y.shape[0] == time_steps_target
    assert inj.shape == (time_steps_target, 2)  # alineada a la longitud real de y
    # Los primeros `injection_length` valores son los reales; el resto es padding con 0.
    assert torch.allclose(inj[:injection_length], torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]]))
    assert torch.all(inj[injection_length:] == 0.0)


@pytest.fixture
def dataset_with_longer_injection(tmp_path):
    """M8: serie de inyeccion mas LARGA que target — debe truncarse."""
    case_dir = tmp_path / "case_001"
    target_dir = case_dir / "layer_cubes"
    target_dir.mkdir(parents=True)
    inj_dir = case_dir / "injection_name_tensors"
    inj_dir.mkdir(parents=True)

    static_names = ["afi_layer_cubes", "cohesion_layer_cubes", "permeability_layer_cubes", "porosity_layer_cubes"]
    for name in static_names:
        (case_dir / name).mkdir(parents=True)

    time_steps_target = 5
    injection_length = 10  # mas larga que target -> debe truncarse

    torch.save(torch.randn(2, time_steps_target, 5, 5), target_dir / "1.pt")
    for name in static_names:
        torch.save(torch.randn(5, 5), case_dir / name / "1.pt")

    values = [round(0.05 * i, 3) for i in range(injection_length)]
    for well in ("tene_1", "tene_2"):
        _write_injection_pt(inj_dir / f"injection_{well}.pt", values)

    return tmp_path, time_steps_target, values


def test_dataset_layers_truncates_long_injection_series(dataset_with_longer_injection):
    root, time_steps_target, values = dataset_with_longer_injection
    ds = DatasetLayers(root, max_layer=time_steps_target - 1)
    x, depth, inj, y = ds[0]

    assert y.shape[0] == time_steps_target
    assert inj.shape == (time_steps_target, 2)
    expected = torch.tensor(values[:time_steps_target])
    assert torch.allclose(inj[:, 0], expected, atol=1e-6)
    assert torch.allclose(inj[:, 1], expected, atol=1e-6)
