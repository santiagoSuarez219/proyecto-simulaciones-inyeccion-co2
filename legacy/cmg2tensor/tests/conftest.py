"""
Shared pytest fixtures for cmg2tensor tests.

Generates synthetic CMG .txt files so tests run without real simulation data.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest


# -------------------------------------------------------------------------
# Helpers to generate synthetic CMG text
# -------------------------------------------------------------------------

def _make_cmg_block(
    nz: int,
    nj: int,
    ni: int,
    time_days: list[int],
    seed: int = 42,
) -> str:
    """
    Generate a synthetic CMG .txt string with TIME / K / J blocks.

    Values are deterministic (seeded random floats in [0, 10]).
    """
    rng = np.random.default_rng(seed)
    lines: list[str] = []

    for t_idx, day in enumerate(time_days):
        year = 2020 + t_idx // 12
        month_names = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                       "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        mon = month_names[t_idx % 12]
        lines.append(f"**  TIME = {day}.0 day  {year}-{mon}-01")
        vmin = float(rng.uniform(0, 1))
        vmax = float(rng.uniform(5, 10))
        lines.append(f"RESULTS PROP  Minimum Value:  {vmin:.4f}  Maximum Value:  {vmax:.4f}")
        for k in range(1, nz + 1):
            for j in range(1, nj + 1):
                vals = rng.uniform(vmin, vmax, ni).astype(np.float32)
                val_str = "  ".join(f"{v:.4f}" for v in vals)
                lines.append(f"**  K =  {k}, J =  {j}")
                lines.append(f"     {val_str}")

    return "\n".join(lines) + "\n"


def _make_static_cmg_block(nz: int, nj: int, ni: int, seed: int = 7) -> str:
    """Synthetic static-snapshot CMG .txt (no TIME headers)."""
    rng = np.random.default_rng(seed)
    lines: list[str] = []
    for k in range(1, nz + 1):
        for j in range(1, nj + 1):
            vals = rng.uniform(0, 1, ni).astype(np.float32)
            val_str = "  ".join(f"{v:.4f}" for v in vals)
            lines.append(f"**  K =  {k}, J =  {j}")
            lines.append(f"     {val_str}")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------
# Grid dimensions used by all tests
# -------------------------------------------------------------------------

NZ, NJ, NI = 2, 3, 4
TIME_DAYS = [365, 730, 1095]   # 3 time steps


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cmg_file(tmp_path_factory) -> Path:
    """A single synthetic CMG .txt file with TIME blocks."""
    p = tmp_path_factory.mktemp("cmg_data") / "SF.txt"
    p.write_text(_make_cmg_block(NZ, NJ, NI, TIME_DAYS), encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def cmg_file_b(tmp_path_factory) -> Path:
    """A second synthetic CMG .txt file (VD) with the same TIME blocks but different values."""
    p = tmp_path_factory.mktemp("cmg_data_b") / "VD.txt"
    p.write_text(_make_cmg_block(NZ, NJ, NI, TIME_DAYS, seed=99), encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def static_cmg_file(tmp_path_factory) -> Path:
    """A static CMG .txt file without TIME blocks."""
    p = tmp_path_factory.mktemp("cmg_static") / "PERM.txt"
    p.write_text(_make_static_cmg_block(NZ, NJ, NI), encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def sim_dir(tmp_path_factory, cmg_file, cmg_file_b) -> Path:
    """A single simulation directory with SF.txt and VD.txt."""
    d = tmp_path_factory.mktemp("sim_001")
    (d / "SF.txt").write_text(cmg_file.read_text(), encoding="utf-8")
    (d / "VD.txt").write_text(cmg_file_b.read_text(), encoding="utf-8")
    return d


@pytest.fixture(scope="session")
def multi_sim_dirs(tmp_path_factory) -> list[Path]:
    """
    10 simulation directories for stress / robustness tests.
    The last one (#10) contains a truncated (corrupt) SF.txt.
    """
    base = tmp_path_factory.mktemp("multi_sims")
    dirs: list[Path] = []
    for i in range(1, 11):
        d = base / f"sim_{i:03d}"
        d.mkdir()
        content_sf = _make_cmg_block(NZ, NJ, NI, TIME_DAYS, seed=i * 10)
        content_vd = _make_cmg_block(NZ, NJ, NI, TIME_DAYS, seed=i * 10 + 1)
        if i == 10:
            # Corrupt: truncate SF.txt mid-file
            content_sf = content_sf[: len(content_sf) // 3]
        (d / "SF.txt").write_text(content_sf, encoding="utf-8")
        (d / "VD.txt").write_text(content_vd, encoding="utf-8")
        dirs.append(d)
    return dirs
