from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fno_co2.etl.histograms import (
    _variable_keys,
    construir_histogramas_globales_por_capas,
    graficar_histograma_global,
)


def _write_layer_cube_npz(
    path: Path,
    cube: np.ndarray,
    variables: list[str],
    *,
    layer_k: int,
) -> None:
    np.savez_compressed(
        path,
        cube=cube.astype(np.float32, copy=False),
        time_ids=np.array([0, 1], dtype=np.int64),
        time_ids_days=np.array([0, 30], dtype=np.int64),
        time_unit=np.array(["months"]),
        variables=np.array(variables),
        layer_k=np.array([layer_k], dtype=np.int64),
        normalization_method=np.array(["minmax"]),
        normalization_applied=np.array([1], dtype=np.int64),
    )


def _write_report(
    report_path: Path,
    output_dir: str,
    variable_order: list[str],
    ranges: dict[str, tuple[float, float]] | None,
) -> None:
    per_variable = {}
    for var_name in variable_order:
        if ranges is None or var_name not in ranges:
            per_variable[var_name] = None
            continue
        vmin, vmax = ranges[var_name]
        per_variable[var_name] = {
            "min": float(vmin),
            "max": float(vmax),
            "span": float(vmax - vmin),
        }

    report = {
        "output_dir": output_dir,
        "variable_order": variable_order,
        "normalization": {
            "applied": ranges is not None,
            "method": "minmax" if ranges is not None else "none",
            "per_variable": per_variable,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _build_processed_dataset(root: Path, *, with_ranges: bool) -> dict[str, np.ndarray]:
    data_by_var: dict[str, list[np.ndarray]] = {"SF": [], "VD": [], "PERMEABILITY": []}

    sim_a = root / "train" / "sim_a"
    sim_a_layers = sim_a / "layer_cubes"
    sim_a_layers.mkdir(parents=True)
    cube_a1 = np.array(
        [
            [
                [[1.0, 2.0], [3.0, np.nan]],
                [[4.0, 5.0], [6.0, 7.0]],
            ],
            [
                [[10.0, 11.0], [12.0, 13.0]],
                [[14.0, 15.0], [16.0, 17.0]],
            ],
        ],
        dtype=np.float32,
    )
    cube_a2 = np.array(
        [
            [
                [[0.5, 1.5], [2.5, 3.5]],
                [[4.5, 5.5], [6.5, 7.5]],
            ],
            [
                [[8.0, 9.0], [10.0, 11.0]],
                [[12.0, 13.0], [14.0, 15.0]],
            ],
        ],
        dtype=np.float32,
    )
    _write_layer_cube_npz(sim_a_layers / "layer_cube_k001.npz", cube_a1, ["SF", "VD"], layer_k=1)
    _write_layer_cube_npz(sim_a_layers / "layer_cube_k002.npz", cube_a2, ["SF", "VD"], layer_k=2)
    sf_a = np.concatenate([cube_a1[0][np.isfinite(cube_a1[0])], cube_a2[0].ravel()])
    vd_a = np.concatenate([cube_a1[1].ravel(), cube_a2[1].ravel()])
    ranges_a = {
        "SF": (float(sf_a.min()), float(sf_a.max())),
        "VD": (float(vd_a.min()), float(vd_a.max())),
    } if with_ranges else None
    _write_report(sim_a / "layer_cubes_report.json", "layer_cubes", ["SF", "VD"], ranges_a)
    data_by_var["SF"].extend([cube_a1[0], cube_a2[0]])
    data_by_var["VD"].extend([cube_a1[1], cube_a2[1]])

    sim_b = root / "train" / "sim_b"
    sim_b_layers = sim_b / "permeability_layer_cubes"
    sim_b_layers.mkdir(parents=True)
    cube_b1 = np.array(
        [
            [
                [[100.0, 150.0], [200.0, 250.0]],
                [[300.0, 350.0], [400.0, 450.0]],
            ]
        ],
        dtype=np.float32,
    )
    cube_b2 = np.array(
        [
            [
                [[125.0, 175.0], [225.0, 275.0]],
                [[325.0, 375.0], [425.0, 475.0]],
            ]
        ],
        dtype=np.float32,
    )
    _write_layer_cube_npz(sim_b_layers / "layer_cube_k001.npz", cube_b1, ["PERMEABILITY"], layer_k=1)
    _write_layer_cube_npz(sim_b_layers / "layer_cube_k002.npz", cube_b2, ["PERMEABILITY"], layer_k=2)
    perm_b = np.concatenate([cube_b1[0].ravel(), cube_b2[0].ravel()])
    ranges_b = {
        "PERMEABILITY": (float(perm_b.min()), float(perm_b.max())),
    } if with_ranges else None
    _write_report(
        sim_b / "permeability_layer_cubes_report.json",
        "permeability_layer_cubes",
        ["PERMEABILITY"],
        ranges_b,
    )
    data_by_var["PERMEABILITY"].extend([cube_b1[0], cube_b2[0]])

    return {
        key: np.concatenate([arr[np.isfinite(arr)] for arr in arrays]).astype(np.float64)
        for key, arrays in data_by_var.items()
    }


def test_construir_histogramas_globales_por_capas_matches_direct_histogram(tmp_path):
    processed_root = tmp_path / "processed"
    direct_values = _build_processed_dataset(processed_root, with_ranges=True)
    output_path = tmp_path / "histograms.json"

    result = construir_histogramas_globales_por_capas(
        dataset=processed_root / "train",
        bins=4,
        output_path=output_path,
    )

    assert output_path.exists()
    assert set(_variable_keys(result)) == {"SF", "VD", "PERMEABILITY"}

    for var_name, values in direct_values.items():
        expected_edges = np.linspace(values.min(), values.max(), 5)
        expected_counts, _ = np.histogram(values, bins=expected_edges)
        assert np.allclose(result[var_name]["bins"], expected_edges)
        assert result[var_name]["counts"] == expected_counts.tolist()
        assert result[var_name]["n"] == int(values.size)
        assert result[var_name]["min"] == float(values.min())
        assert result[var_name]["max"] == float(values.max())
        assert np.isclose(result[var_name]["mean"], values.mean())
        assert np.isclose(result[var_name]["std"], values.std())


def test_construir_histogramas_fallback_to_two_pass_when_report_ranges_missing(tmp_path):
    processed_root = tmp_path / "processed"
    direct_values = _build_processed_dataset(processed_root, with_ranges=False)
    output_path = tmp_path / "histograms_fallback.json"

    result = construir_histogramas_globales_por_capas(
        dataset=processed_root,
        bins=5,
        output_path=output_path,
        variables=["SF", "PERMEABILITY"],
    )

    assert set(_variable_keys(result)) == {"SF", "PERMEABILITY"}
    for var_name in ("SF", "PERMEABILITY"):
        values = direct_values[var_name]
        expected_edges = np.linspace(values.min(), values.max(), 6)
        expected_counts, _ = np.histogram(values, bins=expected_edges)
        assert np.allclose(result[var_name]["bins"], expected_edges)
        assert result[var_name]["counts"] == expected_counts.tolist()


def test_graficar_histograma_global_uses_json_payload(tmp_path):
    payload = {
        "SF": {
            "bins": [0.0, 1.0, 2.0],
            "counts": [3, 7],
            "min": 0.0,
            "max": 2.0,
            "mean": 1.2,
            "std": 0.4,
            "n": 10,
            "p5": 0.1,
            "p50": 1.1,
            "p95": 1.9,
        }
    }
    json_path = tmp_path / "payload.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ModuleNotFoundError:
        return

    ax = graficar_histograma_global(json_path, variable="SF", log_y=True)
    assert ax.get_title() == "Histograma global | SF"
    assert ax.get_yscale() == "log"
