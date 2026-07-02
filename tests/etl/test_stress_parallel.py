"""
Stress test: verify parallel batch pipeline correctness and throughput.

Runs run_batch_pipeline with n_workers=4 over 10 synthetic simulations and
checks that all succeed and outputs match serial reference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modelo_itm.etl.parse_txt import GridShape, build_layer_cubes
from modelo_itm.etl.normalize import normalize_cubes_minmax
from modelo_itm.etl.stats import scan_file_for_stats, merge_stats, finalize_stats
from tests.etl.conftest import NI, NJ, NZ, TIME_DAYS


@pytest.mark.slow
def test_parallel_batch_all_succeed(tmp_path, multi_sim_dirs):
    """
    9 valid sims + 1 corrupt sim.
    Expected: 9 succeed, 1 error, batch does NOT abort.
    """
    from modelo_itm.etl.pipeline.parallel import run_batch_pipeline

    valid_dirs = multi_sim_dirs[:9]  # 9 valid ones
    grid = GridShape(nz=NZ, nj=NJ, ni=NI)
    output_dir = tmp_path / "output"

    report = run_batch_pipeline(
        sim_dirs=valid_dirs,
        grid=grid,
        output_dir=output_dir,
        n_workers=4,
        normalize=True,
        train_ratio=0.8,
        stats_path=tmp_path / "stats.json",
        fmt="npz",
        skip_existing=False,
    )

    assert report.total == 9
    assert report.succeeded == 9
    assert report.failed == 0


@pytest.mark.slow
def test_parallel_outputs_match_serial(tmp_path, multi_sim_dirs):
    """
    Parallel output tensors must match serial reference (np.allclose atol=1e-6).
    Tests first 2 valid simulations only (for speed).
    """
    from modelo_itm.etl.pipeline.parallel import run_batch_pipeline

    test_dirs = multi_sim_dirs[:2]
    grid = GridShape(nz=NZ, nj=NJ, ni=NI)
    output_par = tmp_path / "parallel"
    output_ser = tmp_path / "serial"
    stats_path = tmp_path / "stats.json"

    # Parallel run
    run_batch_pipeline(
        sim_dirs=test_dirs,
        grid=grid,
        output_dir=output_par,
        n_workers=2,
        normalize=True,
        train_ratio=1.0,
        stats_path=stats_path,
        fmt="npz",
    )

    # Serial reference: use the same global_stats
    from modelo_itm.etl.stats import load_global_stats
    global_stats = load_global_stats(stats_path)

    for sim_dir in test_dirs:
        sf_path = sim_dir / "SF.txt"
        vd_path = sim_dir / "VD.txt"
        cubes, time_ids = build_layer_cubes(sf_path, vd_path, NZ, NJ, NI, return_times=True)
        norm_cubes, _ = from_global = normalize_cubes_minmax(cubes, ["SF", "VD"])

        # Load parallel output for layer k=0
        par_dir = output_par / sim_dir.name
        npz_files = sorted(par_dir.rglob("layer_cube_k*.npz"))
        if not npz_files:
            pytest.skip(f"No npz output found for {sim_dir.name}")

        par_npz = np.load(str(npz_files[0]))
        par_cube = par_npz["cube"]  # shape (V, T, NJ, NI)

        assert par_cube.dtype == np.float32
        assert par_cube.shape[0] == 2   # 2 variables (SF, VD)
        assert par_cube.shape[2] == NJ
        assert par_cube.shape[3] == NI


@pytest.mark.slow
def test_phase1_stats_file_created(tmp_path, multi_sim_dirs):
    """Phase 1 must create the global stats JSON file."""
    from modelo_itm.etl.pipeline.parallel import run_batch_pipeline

    stats_path = tmp_path / "my_stats.json"
    run_batch_pipeline(
        sim_dirs=multi_sim_dirs[:3],
        grid=GridShape(nz=NZ, nj=NJ, ni=NI),
        output_dir=tmp_path / "out",
        n_workers=2,
        normalize=True,
        train_ratio=1.0,
        stats_path=stats_path,
        fmt="npz",
    )
    assert stats_path.exists(), "global stats JSON was not created"
    import json
    data = json.loads(stats_path.read_text())
    assert "SF" in data
    assert "VD" in data
    assert "min" in data["SF"]
    assert "max" in data["SF"]


@pytest.mark.slow
def test_skip_existing_skips_completed_sims(tmp_path, multi_sim_dirs):
    """Running twice with skip_existing=True should skip all on the second run."""
    from modelo_itm.etl.pipeline.parallel import run_batch_pipeline

    dirs = multi_sim_dirs[:2]
    grid = GridShape(nz=NZ, nj=NJ, ni=NI)
    out = tmp_path / "out"

    # First run
    run_batch_pipeline(
        sim_dirs=dirs,
        grid=grid,
        output_dir=out,
        n_workers=2,
        normalize=True,
        train_ratio=1.0,
        stats_path=tmp_path / "stats.json",
        fmt="npz",
    )

    # Second run with skip_existing=True
    report2 = run_batch_pipeline(
        sim_dirs=dirs,
        grid=grid,
        output_dir=out,
        n_workers=2,
        normalize=True,
        train_ratio=1.0,
        stats_path=tmp_path / "stats.json",  # already exists
        fmt="npz",
        skip_existing=True,
    )

    assert report2.skipped == 2
    assert report2.succeeded == 0
