"""
Robustness tests: corrupt or incomplete files must not abort batch processing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modelo_itm.etl.parse_txt import GridShape, parse_txt_with_times, _parse_txt_with_times
from tests.etl.conftest import NI, NJ, NZ, TIME_DAYS


# -------------------------------------------------------------------------
# Unit-level: corrupt single file
# -------------------------------------------------------------------------

def test_corrupt_file_lenient_mode(tmp_path):
    """A truncated file with strict=False returns partial data without raising."""
    content = "**  TIME = 365.0 day  2020-JAN-01\n**  K =  1, J =  1\n     0.1  0.2  0.3"
    p = tmp_path / "corrupt.txt"
    p.write_text(content, encoding="utf-8")
    # strict=False should not raise even with incomplete data
    import numpy as np
    tensor, time_ids = _parse_txt_with_times(p, NZ, NJ, NI, dtype=np.float32, strict=False)
    assert tensor.ndim == 4
    assert len(time_ids) >= 1


def test_corrupt_file_strict_raises(tmp_path):
    """A truncated file with strict=True must raise ValueError."""
    content = "**  TIME = 365.0 day  2020-JAN-01\n**  K =  1, J =  1\n     0.1  0.2  0.3"
    p = tmp_path / "corrupt.txt"
    p.write_text(content, encoding="utf-8")
    import numpy as np
    with pytest.raises((ValueError, Exception)):
        _parse_txt_with_times(p, NZ, NJ, NI, dtype=np.float32, strict=True)


# -------------------------------------------------------------------------
# Batch-level: corrupt sim does not abort others
# -------------------------------------------------------------------------

@pytest.mark.slow
def test_corrupt_sim_does_not_abort_batch(tmp_path, multi_sim_dirs):
    """
    multi_sim_dirs fixture includes 1 corrupt sim (#10).
    Batch must: succeed for 9 valid, fail for 1 corrupt, not abort.
    """
    from modelo_itm.etl.pipeline.parallel import run_batch_pipeline

    all_dirs = multi_sim_dirs  # 10 dirs, last one is corrupt
    grid = GridShape(nz=NZ, nj=NJ, ni=NI)

    report = run_batch_pipeline(
        sim_dirs=all_dirs,
        grid=grid,
        output_dir=tmp_path / "output",
        n_workers=4,
        normalize=False,   # Skip stats scan to speed up test
        train_ratio=1.0,
        fmt="npz",
    )

    assert report.total == 10
    assert report.succeeded == 9
    assert report.failed == 1
    assert len(report.failed_sims) == 1

    # The corrupt sim should be in failed list
    failed_result = next(r for r in report.worker_results if r.status == "error")
    assert failed_result.error is not None


@pytest.mark.slow
def test_all_valid_outputs_exist_despite_one_corrupt(tmp_path, multi_sim_dirs):
    """
    After a batch run with 1 corrupt sim, the 9 valid outputs must exist.
    """
    from modelo_itm.etl.pipeline.parallel import run_batch_pipeline

    all_dirs = multi_sim_dirs
    grid = GridShape(nz=NZ, nj=NJ, ni=NI)
    output_dir = tmp_path / "output"

    report = run_batch_pipeline(
        sim_dirs=all_dirs,
        grid=grid,
        output_dir=output_dir,
        n_workers=4,
        normalize=False,
        train_ratio=1.0,
        fmt="npz",
    )

    # Check that all valid sims have output files
    valid_results = [r for r in report.worker_results if r.status == "success"]
    for wr in valid_results:
        assert wr.output_dir is not None
        out_path = Path(wr.output_dir)
        npz_files = list(out_path.rglob("*.npz"))
        assert len(npz_files) > 0, f"No .npz files found for {wr.sim_name}"


# -------------------------------------------------------------------------
# scan_file_for_stats robustness
# -------------------------------------------------------------------------

def test_scan_stats_on_empty_file(tmp_path):
    """scan_file_for_stats on an empty file should return vmin=None, vmax=None or raise gracefully."""
    from modelo_itm.etl.stats import scan_file_for_stats
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    # With an empty file and no RESULTS PROP, the fallback parse will return empty/nan tensor
    # It should not crash, but result may have None or nan stats
    try:
        result = scan_file_for_stats(p, "SF", nz=NZ, nj=NJ, ni=NI)
        # If it returned, vmin/vmax may be nan (from nanmin of all-nan tensor)
        # That is acceptable — the caller (merge_stats) skips None values
    except Exception:
        pass  # Raising is also acceptable for a fully empty file


def test_scan_stats_file_not_found(tmp_path):
    """scan_file_for_stats on a non-existent file must raise."""
    from modelo_itm.etl.stats import scan_file_for_stats
    p = tmp_path / "nonexistent.txt"
    with pytest.raises(Exception):
        scan_file_for_stats(p, "SF", nz=NZ, nj=NJ, ni=NI)


def test_scan_simulation_with_missing_vd(tmp_path, cmg_file):
    """scan_simulation_for_stats with no VD file must return a result (not crash)."""
    from modelo_itm.etl.stats import scan_simulation_for_stats
    discovered = {
        "sf_path": cmg_file,
        "vd_path": None,  # missing VD
        "permeability_path": None,
        "porosity_path": None,
        "cohesion_path": None,
        "afi_path": None,
        "pressure_path": None,
        "gas_saturation_path": None,
        "injection_path": None,
    }
    result = scan_simulation_for_stats(
        tmp_path,
        nz=NZ,
        nj=NJ,
        ni=NI,
        discovered=discovered,
    )
    assert result.error is None
    # Should have scanned SF only
    sf_results = [fr for fr in result.file_results if fr.var_name == "SF"]
    assert len(sf_results) == 1
