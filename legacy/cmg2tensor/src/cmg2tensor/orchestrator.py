"""
Batch orchestration utilities for cmg2tensor.

Responsibilities:
  - _load_train_test_split_csv     — parse split CSV → {name: split}
  - _write_batch_report_incremental — write/update batch JSON report
  - _compute_global_minmax_for_simulations — serial Phase 1 (global stats)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .discovery import discover_simulation_inputs
from .stats import (
    scan_simulation_for_stats,
    merge_stats,
    finalize_stats,
)


def _load_train_test_split_csv(path: Path) -> dict[str, str]:
    """
    Parse a split CSV with columns [simulation_name, split].

    Returns {simulation_name: 'train'|'test'}.
    Returns {} if the file does not exist or lacks the required columns.
    """
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        if "simulation_name" not in reader.fieldnames or "split" not in reader.fieldnames:
            return {}
        for row in reader:
            name = str(row.get("simulation_name", "")).strip()
            split = str(row.get("split", "")).strip().lower()
            if not name or split not in {"train", "test"}:
                continue
            mapping[name] = split
    return mapping


def _write_batch_report_incremental(
    *,
    report_path: Path,
    raw_root: Path,
    processed_root: Path,
    simulation_dirs: list[Path],
    simulations_report: dict[str, Any],
    succeeded_simulations: list[str],
    existing_outputs_simulations: list[str],
    skipped_simulations: list[str],
    failed_simulations: list[str],
    start_total: float,
) -> None:
    elapsed_seconds = round(perf_counter() - start_total, 6)
    batch_report = {
        "mode": "multi-simulation",
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "simulations_count": len(simulation_dirs),
        "simulations_succeeded": list(succeeded_simulations),
        "simulations_existing_outputs": list(existing_outputs_simulations),
        "simulations_skipped": list(skipped_simulations),
        "simulations_failed": list(failed_simulations),
        "simulations": simulations_report,
        "execution_seconds_total": elapsed_seconds,
        "incremental": True,
    }
    report_path.write_text(json.dumps(batch_report, indent=2), encoding="utf-8")


def _compute_global_minmax_for_simulations(
    sim_dirs: list[Path],
    *,
    nz: int,
    nj: int,
    ni: int,
    injection_sheet: str,
) -> dict[str, dict[str, float]]:
    """
    Compute global min/max per variable across the given simulations (serial).

    Delegates to stats.scan_simulation_for_stats — line-by-line scan with
    RESULTS PROP fast path (O(1) memory). See pipeline/parallel.py for the
    parallel version which does the same work with ProcessPoolExecutor.

    Variables covered: SF, VD, PERMEABILITY, POROSITY, COHESION, AFI,
                       PRESSURE, GAS_SATURATION, TENE-1, TENE-2 (if present).
    """
    stats: dict[str, dict[str, float]] = {}
    for sim_dir in sim_dirs:
        try:
            discovered = discover_simulation_inputs(sim_dir)
        except ValueError:
            continue

        if discovered["sf_path"] is None or discovered["vd_path"] is None:
            continue

        sim_result = scan_simulation_for_stats(
            sim_dir,
            nz=nz, nj=nj, ni=ni,
            discovered=discovered,
            injection_sheet=injection_sheet,
        )
        stats = merge_stats(stats, sim_result)

    return finalize_stats(stats)
