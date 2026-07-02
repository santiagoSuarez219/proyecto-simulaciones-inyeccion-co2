"""
Two-phase parallel batch pipeline for cmg2tensor.

Phase 1: Compute global min/max statistics across train simulations.
Phase 2: Parse, normalize, and save all simulations in parallel.

All worker functions are module-level (not closures or lambdas) to ensure they
are picklable by ProcessPoolExecutor on Windows (spawn context).

Usage
-----
from modelo_itm.etl.pipeline import run_batch_pipeline, BatchReport
from modelo_itm.etl.parse_txt import GridShape
from pathlib import Path

report = run_batch_pipeline(
    sim_dirs=sorted(Path("data/raw").iterdir()),
    grid=GridShape(nz=20, nj=100, ni=100),
    output_dir=Path("data/processed"),
    n_workers=8,
    normalize=True,
    fmt="pt",
)
print(f"Succeeded: {report.succeeded}/{report.total}")
print(f"Phase 1: {report.phase1_seconds:.1f}s  Phase 2: {report.phase2_seconds:.1f}s")
"""
from __future__ import annotations

import os
import time
import traceback as _traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..parse_txt import GridShape
from ..stats import (
    SimulationScanResult,
    scan_simulation_for_stats,
    merge_stats,
    finalize_stats,
    save_global_stats,
    load_global_stats,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    """Result returned by a single Phase 2 transform worker."""

    sim_name: str
    status: str                      # "success" | "error" | "skipped"
    output_dir: str | None = None
    output_files: list[str] = field(default_factory=list)
    error: str | None = None
    tb: str | None = None            # full traceback for failed workers
    execution_seconds: float = 0.0


@dataclass
class BatchReport:
    """Aggregate report from run_batch_pipeline."""

    total: int
    succeeded: int
    failed: int
    skipped: int
    failed_sims: list[str]
    output_dir: Path
    stats_path: Path | None
    phase1_seconds: float
    phase2_seconds: float
    worker_results: list[WorkerResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "failed_sims": self.failed_sims,
            "output_dir": str(self.output_dir),
            "stats_path": str(self.stats_path) if self.stats_path else None,
            "phase1_seconds": self.phase1_seconds,
            "phase2_seconds": self.phase2_seconds,
            "worker_results": [
                {
                    "sim_name": r.sim_name,
                    "status": r.status,
                    "output_dir": r.output_dir,
                    "output_files_count": len(r.output_files),
                    "error": r.error,
                    "execution_seconds": r.execution_seconds,
                }
                for r in self.worker_results
            ],
        }


# ---------------------------------------------------------------------------
# Phase 1 worker — stats scan
# ---------------------------------------------------------------------------

def _scan_stats_worker(
    args: tuple[Path, int, int, int, dict[str, Any], str],
) -> SimulationScanResult:
    """
    Phase 1 top-level worker function.

    Scans one simulation directory for per-variable min/max statistics.
    Uses line-by-line scanning (RESULTS PROP headers) when available;
    falls back to full streaming parse + immediate del for the rest.

    Parameters (packed as tuple for ProcessPoolExecutor.map):
        sim_dir, nz, nj, ni, discovered, injection_sheet
    """
    sim_dir, nz, nj, ni, discovered, injection_sheet = args
    return scan_simulation_for_stats(
        sim_dir,
        nz=nz,
        nj=nj,
        ni=ni,
        discovered=discovered,
        injection_sheet=injection_sheet,
    )


# ---------------------------------------------------------------------------
# Phase 2 worker — parse + normalize + save
# ---------------------------------------------------------------------------

def _transform_worker(
    args: tuple[
        Path,          # sim_dir
        GridShape,     # grid
        dict,          # global_stats (or {} for no normalization)
        Path,          # output_dir (base; subdir per sim)
        str,           # fmt: "pt" | "npz"
        bool,          # normalize
        bool,          # skip_existing
        dict,          # discovered inputs
        str,           # injection_sheet
        bool,          # full_tensor_output
        str | None,    # split: "train" | "test" | None
    ],
) -> WorkerResult:
    """
    Phase 2 top-level worker function.

    Parses all variable files for one simulation, normalizes using global_stats,
    saves .pt/.npz layer cubes, then explicitly releases RAM.

    Returns a WorkerResult with status "success", "skipped", or "error".
    Failed workers do NOT raise — they capture the exception and return
    WorkerResult(status="error") so the orchestrator can continue with other sims.
    """
    (
        sim_dir, grid, global_stats, output_dir, fmt, normalize,
        skip_existing, discovered, injection_sheet, full_tensor_output, split,
    ) = args

    t0 = time.perf_counter()
    sim_name = sim_dir.name

    try:
        # Determine per-simulation processed directory
        if split is not None:
            sim_processed_dir = output_dir / split / sim_name
        else:
            sim_processed_dir = output_dir / sim_name

        sim_processed_dir.mkdir(parents=True, exist_ok=True)

        # Skip if existing report present and skip_existing is set
        report_path = sim_processed_dir / "layer_cubes_report.json"
        if skip_existing and report_path.exists():
            return WorkerResult(
                sim_name=sim_name,
                status="skipped",
                output_dir=str(sim_processed_dir),
                execution_seconds=time.perf_counter() - t0,
            )

        # Import here (inside worker process) to avoid heavyweight imports at spawn
        from ..pipelines import _run_requested_pipelines  # noqa: PLC0415

        torch_output = fmt == "pt"
        # Test simulations are never normalized (stats are train-only)
        normalize_this = normalize and split != "test"
        global_norm: dict | None = global_stats if (normalize_this and global_stats) else None

        reports = _run_requested_pipelines(
            sf_path=discovered.get("sf_path"),
            vd_path=discovered.get("vd_path"),
            permeability_path=discovered.get("permeability_path"),
            porosity_path=discovered.get("porosity_path"),
            cohesion_path=discovered.get("cohesion_path"),
            afi_path=discovered.get("afi_path"),
            pressure_path=discovered.get("pressure_path"),
            gas_saturation_path=discovered.get("gas_saturation_path"),
            injection_path=discovered.get("injection_path"),
            injection_sheet=injection_sheet,
            processed_dir=sim_processed_dir,
            nz=grid.nz,
            nj=grid.nj,
            ni=grid.ni,
            normalize=normalize_this,
            full_tensor_output=full_tensor_output,
            torch_output=torch_output,
            global_stats=global_norm,
        )

        del reports

        # Collect output file list from saved layer cubes
        suffix = ".pt" if torch_output else ".npz"
        output_files = [str(p) for p in sim_processed_dir.rglob(f"*{suffix}")]

        return WorkerResult(
            sim_name=sim_name,
            status="success",
            output_dir=str(sim_processed_dir),
            output_files=output_files,
            execution_seconds=time.perf_counter() - t0,
        )

    except Exception as exc:  # noqa: BLE001
        return WorkerResult(
            sim_name=sim_name,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            tb=_traceback.format_exc(),
            execution_seconds=time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_batch_pipeline(
    sim_dirs: list[Path],
    grid: GridShape,
    output_dir: Path,
    *,
    n_workers: int | None = None,
    normalize: bool = True,
    train_ratio: float = 0.8,
    split_map: dict[str, str] | None = None,
    stats_path: Path | None = None,
    fmt: Literal["pt", "npz"] = "pt",
    skip_existing: bool = False,
    retry_failed: list[str] | None = None,
    injection_sheet: str = "Well Summary",
    full_tensor_output: bool = False,
) -> BatchReport:
    """
    Two-phase parallel batch pipeline.

    Phase 1 — Global stats (train simulations only):
        ProcessPoolExecutor scans each train sim for per-variable min/max.
        Uses RESULTS PROP header scan (O(1) memory) when available;
        falls back to streaming parse + immediate del otherwise.
        Saves result to stats_path (if provided) as global_stats.json.

    Phase 2 — Transform all simulations:
        ProcessPoolExecutor parses + normalizes + saves each sim.
        Each worker is an isolated process; a failing sim returns
        WorkerResult(status="error") without aborting the others.

    Parameters
    ----------
    sim_dirs:
        All simulation input directories (train + test combined).
    grid:
        GridShape(nz, nj, ni).
    output_dir:
        Root output directory. Sims go to output_dir/<sim_name>/ or
        output_dir/train/<sim_name>/ / output_dir/test/<sim_name>/ when
        split_map is provided.
    n_workers:
        Number of parallel workers. Defaults to min(8, cpu_count).
    normalize:
        Apply min-max normalization.
    train_ratio:
        Fraction of sim_dirs to use as train when split_map is None.
    split_map:
        Optional mapping sim_name → "train" | "test" (from --split-csv).
        When None, the first train_ratio fraction of sim_dirs is treated as train.
    stats_path:
        Path to save/load global_stats.json. If the file exists, Phase 1 is skipped.
    fmt:
        Output format: "pt" (PyTorch) or "npz" (NumPy).
    skip_existing:
        Skip simulations whose layer_cubes_report.json already exists.
    retry_failed:
        If provided, only process the named simulations (Phase 2 only).
        Useful to reprocess failed sims from a previous run.
    injection_sheet:
        Excel sheet name for injection data.
    full_tensor_output:
        Save full (V,T,Z,J,I) tensor instead of per-layer files.
    """
    import multiprocessing

    if n_workers is None:
        n_workers = min(8, os.cpu_count() or 1)

    # Windows-safe: always use spawn context
    ctx = multiprocessing.get_context("spawn")

    # Discover inputs for ALL sims upfront (fast; done in main process)
    from ..discovery import discover_simulation_inputs  # noqa: PLC0415

    discovered_map: dict[str, dict[str, Any]] = {}
    valid_sim_dirs: list[Path] = []
    for sim_dir in sim_dirs:
        try:
            discovered = discover_simulation_inputs(sim_dir)
        except ValueError:
            continue
        discovered_map[sim_dir.name] = discovered
        valid_sim_dirs.append(sim_dir)

    # Determine train/test split
    if split_map:
        train_dirs = [d for d in valid_sim_dirs if split_map.get(d.name) == "train"]
    else:
        n_train = max(1, int(len(valid_sim_dirs) * train_ratio))
        train_dirs = valid_sim_dirs[:n_train]

    # If retry_failed provided, restrict Phase 2 to those sims
    if retry_failed:
        retry_set = set(retry_failed)
        phase2_dirs = [d for d in valid_sim_dirs if d.name in retry_set]
    else:
        phase2_dirs = valid_sim_dirs

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: compute global stats
    # -----------------------------------------------------------------------
    t1_start = time.perf_counter()
    global_stats: dict[str, dict[str, float]] = {}

    if stats_path and stats_path.exists():
        global_stats = load_global_stats(stats_path)
        print(f"[PHASE1] Loaded existing stats from {stats_path}")
    elif normalize and train_dirs:
        print(f"[PHASE1] Scanning {len(train_dirs)} train simulations with {n_workers} workers …")
        phase1_args = [
            (d, grid.nz, grid.nj, grid.ni, discovered_map[d.name], injection_sheet)
            for d in train_dirs
        ]
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            for sim_result in pool.map(_scan_stats_worker, phase1_args):
                global_stats = merge_stats(global_stats, sim_result)
                if sim_result.error:
                    print(f"[PHASE1] WARNING: scan failed for '{sim_result.sim_name}': {sim_result.error.splitlines()[0]}")
        global_stats = finalize_stats(global_stats)
        if stats_path:
            save_global_stats(global_stats, stats_path)
            print(f"[PHASE1] Stats saved to {stats_path}")
    else:
        print("[PHASE1] Skipped (normalize=False or no train dirs).")

    phase1_seconds = time.perf_counter() - t1_start
    print(f"[PHASE1] Done in {phase1_seconds:.1f}s")

    # -----------------------------------------------------------------------
    # Phase 2: transform all simulations
    # -----------------------------------------------------------------------
    t2_start = time.perf_counter()
    print(f"[PHASE2] Transforming {len(phase2_dirs)} simulations with {n_workers} workers …")

    phase2_args = [
        (
            d,
            grid,
            global_stats,
            output_dir,
            fmt,
            normalize,
            skip_existing,
            discovered_map[d.name],
            injection_sheet,
            full_tensor_output,
            split_map.get(d.name) if split_map else None,
        )
        for d in phase2_dirs
    ]

    worker_results: list[WorkerResult] = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_transform_worker, args): args[0].name for args in phase2_args}
        for future in as_completed(futures):
            result = future.result()
            worker_results.append(result)
            status_symbol = "✓" if result.status == "success" else ("–" if result.status == "skipped" else "✗")
            print(f"  [{status_symbol}] {result.sim_name} ({result.execution_seconds:.1f}s)", flush=True)
            if result.status == "error":
                print(f"      ERROR: {result.error}", flush=True)

    phase2_seconds = time.perf_counter() - t2_start

    succeeded, failed, skipped_results = [], [], []
    for r in worker_results:
        if r.status == "success":
            succeeded.append(r)
        elif r.status == "error":
            failed.append(r)
        else:
            skipped_results.append(r)

    print(
        f"[PHASE2] Done in {phase2_seconds:.1f}s — "
        f"{len(succeeded)} ok / {len(failed)} failed / {len(skipped_results)} skipped"
    )

    return BatchReport(
        total=len(phase2_dirs),
        succeeded=len(succeeded),
        failed=len(failed),
        skipped=len(skipped_results),
        failed_sims=[r.sim_name for r in failed],
        output_dir=output_dir,
        stats_path=stats_path,
        phase1_seconds=phase1_seconds,
        phase2_seconds=phase2_seconds,
        worker_results=worker_results,
    )
