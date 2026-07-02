"""
C1 — verifies that the "test" split IS normalized with the same train-only
global stats, instead of being written in raw physical units.

Before the fix: normalize_this = normalize and split != "test" forced test
outputs to stay unnormalized, while train was scaled to [0, 1] — making
val_loss/val_r2/val_rmse incomparable with train.
"""
from __future__ import annotations

import numpy as np
import pytest

from fno_co2.etl.parse_txt import GridShape
from tests.etl.conftest import NI, NJ, NZ


@pytest.mark.slow
def test_test_split_is_normalized_with_train_stats(tmp_path, multi_sim_dirs):
    from fno_co2.etl.pipeline.parallel import run_batch_pipeline
    from fno_co2.etl.stats import load_global_stats

    # 6 train + 2 test (skip the 2 remaining, one of which is corrupt)
    dirs = multi_sim_dirs[:8]
    split_map = {d.name: ("train" if i < 6 else "test") for i, d in enumerate(dirs)}

    grid = GridShape(nz=NZ, nj=NJ, ni=NI)
    output_dir = tmp_path / "processed"
    stats_path = tmp_path / "stats.json"

    run_batch_pipeline(
        sim_dirs=dirs,
        grid=grid,
        output_dir=output_dir,
        n_workers=2,
        normalize=True,
        split_map=split_map,
        stats_path=stats_path,
        fmt="npz",
    )

    global_stats = load_global_stats(stats_path)
    assert "SF" in global_stats and "VD" in global_stats

    train_sim = next(d for d in dirs if split_map[d.name] == "train")
    test_sim = next(d for d in dirs if split_map[d.name] == "test")

    train_files = sorted((output_dir / "train" / train_sim.name).rglob("layer_cube_k*.npz"))
    test_files = sorted((output_dir / "test" / test_sim.name).rglob("layer_cube_k*.npz"))
    assert train_files, "no train outputs were produced"
    assert test_files, "no test outputs were produced — split routing broke Phase 2"

    def _values_in_unit_range(npz_path):
        cube = np.load(str(npz_path))["cube"]
        return float(cube.min()), float(cube.max())

    train_min, train_max = _values_in_unit_range(train_files[0])
    test_min, test_max = _values_in_unit_range(test_files[0])

    # Normalized data must be within [0, 1] (min-max scaling); raw physical
    # units (the pre-fix bug) would very likely fall outside this range for
    # synthetic values seeded in [0, 10] by the conftest generators.
    assert -1e-6 <= train_min and train_max <= 1.0 + 1e-6, (
        f"train not normalized: min={train_min}, max={train_max}"
    )
    assert -1e-6 <= test_min and test_max <= 1.0 + 1e-6, (
        f"test split was NOT normalized (C1 regression): min={test_min}, max={test_max}"
    )


@pytest.mark.slow
def test_test_split_report_reflects_normalization(tmp_path, multi_sim_dirs):
    """layer_cubes_report.json for a test-split sim must record normalize=True,
    proving Phase 2 actually applied the train stats instead of silently
    skipping normalization for that split."""
    from fno_co2.etl.pipeline.parallel import run_batch_pipeline

    dirs = multi_sim_dirs[:4]
    split_map = {d.name: ("train" if i < 2 else "test") for i, d in enumerate(dirs)}

    grid = GridShape(nz=NZ, nj=NJ, ni=NI)
    output_dir = tmp_path / "processed"

    run_batch_pipeline(
        sim_dirs=dirs,
        grid=grid,
        output_dir=output_dir,
        n_workers=2,
        normalize=True,
        split_map=split_map,
        stats_path=tmp_path / "stats.json",
        fmt="npz",
    )

    test_sim = next(d for d in dirs if split_map[d.name] == "test")
    report_path = output_dir / "test" / test_sim.name / "layer_cubes_report.json"
    assert report_path.exists()

    import json
    report = json.loads(report_path.read_text())
    normalization = report.get("normalization", {})
    assert normalization.get("applied") is True, (
        f"test split report says normalization.applied={normalization.get('applied')!r} "
        "— C1 not fixed (test outputs were not normalized)"
    )
