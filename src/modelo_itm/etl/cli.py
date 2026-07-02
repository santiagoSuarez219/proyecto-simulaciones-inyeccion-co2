"""
CLI entry point for cmg2tensor.

Contains _build_arg_parser() and main().
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import (
    DEFAULT_BATCH_REPORT_PATH,
    DEFAULT_GLOBAL_NORMALIZATION_DIR,
    DEFAULT_GLOBAL_STATS_FILE,
)
from .orchestrator import (
    _load_train_test_split_csv,
    _write_batch_report_incremental,
    _compute_global_minmax_for_simulations,
)
from .parse_txt import GridShape
from .pipeline.serial import _run_requested_pipelines, _print_execution_time
from .stats import save_global_stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelo_itm.etl",
        description="Build per-layer cubes (V, T, NJ, NI) from CMG text files.",
    )
    parser.add_argument(
        "--processed-dir", "--output-dir",
        dest="processed_dir",
        default="data/processed",
        help="Directory for outputs and execution reports. Created automatically if it does not exist.",
    )
    parser.add_argument(
        "--all-simulations",
        action="store_true",
        help=(
            "Run one pipeline per simulation folder inside --raw-root. "
            "Each subfolder is treated as one simulation; outputs are written to "
            "<processed-dir>/<simulation_name>/."
        ),
    )
    parser.add_argument(
        "--skip-missing-required",
        action="store_true",
        help=(
            "Only for --all-simulations. Skip folders missing required inputs "
            "(SF/VD) instead of aborting."
        ),
    )
    parser.add_argument(
        "--skip-existing-outputs",
        action="store_true",
        help=(
            "Only for --all-simulations. Skip simulations whose "
            "<processed-dir>/<sim>/layer_cubes_report.json already exists."
        ),
    )
    parser.add_argument(
        "--skip-existing-pipelines",
        action="store_true",
        help=(
            "Skip any pipeline whose report already exists inside the simulation output folder."
        ),
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help=(
            "Optional train/test split CSV with columns [simulation_name, split]. "
            "When provided, batch outputs go to <processed-dir>/<split>/<simulation_name>/. "
            "Normalization is applied only to split=train; test is saved without normalization."
        ),
    )
    parser.add_argument(
        "--normalization-scope",
        choices=("simulation", "split"),
        default="split",
        help=(
            "'simulation' normalizes each sim independently (legacy). "
            "'split' computes global min/max from train and applies it."
        ),
    )
    parser.add_argument(
        "--global-normalization-stats",
        type=Path,
        default=None,
        help=(
            "Path to an existing global stats JSON (train.json). "
            "When provided, Phase 1 is skipped and these stats are reused."
        ),
    )
    parser.add_argument(
        "--compute-global-normalization-only",
        action="store_true",
        help=(
            "Compute global min/max for split=train, write to "
            "reports/global_normalization/train.json, then exit."
        ),
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Root folder containing simulation subfolders for --all-simulations.",
    )
    parser.add_argument(
        "--shared-permeability-path",
        default=None,
        help="Fallback permeability file when a simulation folder has none.",
    )
    parser.add_argument(
        "--shared-porosity-path",
        default=None,
        help="Fallback porosity file when a simulation folder has none.",
    )
    parser.add_argument("--sf-path", default=None, help="SF file for single-simulation mode.")
    parser.add_argument("--vd-path", default=None, help="VD file for single-simulation mode.")
    parser.add_argument("--permeability-path", default=None)
    parser.add_argument("--porosity-path", default=None)
    parser.add_argument("--cohesion-path", default=None)
    parser.add_argument("--afi-path", default=None)
    parser.add_argument("--pressure-path", default=None)
    parser.add_argument("--gas-saturation-path", default=None)
    parser.add_argument(
        "--injection-path", default=None,
        help="Excel file with columns Time (day), Name, Value.",
    )
    parser.add_argument(
        "--injection-sheet", default="Well Summary",
        help="Sheet name to read from --injection-path.",
    )
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--nj", type=int, default=100)
    parser.add_argument("--ni", type=int, default=100)

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--torch-output", dest="torch_output", action="store_true",
                               help="Save as .pt (default).")
    output_group.add_argument("--npz-output", dest="torch_output", action="store_false",
                               help="Save as .npz.")
    parser.set_defaults(torch_output=True)

    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument("--normalize", dest="normalize", action="store_true",
                                  help="Apply min-max normalization (default).")
    normalize_group.add_argument("--no-normalize", dest="normalize", action="store_false",
                                  help="Save raw values without normalization.")
    parser.set_defaults(normalize=True)

    parser.add_argument(
        "--full-tensor-output", action="store_true",
        help="Save one full (V, T, Z, J, I) tensor per variable instead of per-layer files.",
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False,
        help="Enable parallel batch processing with ProcessPoolExecutor.",
    )
    parser.add_argument(
        "--n-workers", dest="n_workers", type=int, default=None, metavar="N",
        help="Number of parallel workers (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--retry-failed", dest="retry_failed", default=None, metavar="REPORT_JSON",
        help="Path to a previous batch_parallel_report.json. Re-runs failed simulations only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from .discovery import discover_simulation_inputs  # noqa: PLC0415

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    start_total = perf_counter()
    try:
        if args.all_simulations:
            manual_inputs = [
                args.sf_path, args.vd_path, args.permeability_path, args.porosity_path,
                args.cohesion_path, args.afi_path, args.pressure_path,
                args.gas_saturation_path, args.injection_path,
            ]
            if any(path is not None for path in manual_inputs):
                raise ValueError(
                    "Do not mix --all-simulations with manual input paths. "
                    "Use either simulation-folder discovery OR explicit file paths."
                )

            raw_root = Path(args.raw_root)
            if not raw_root.exists():
                raise ValueError(f"--raw-root does not exist: {raw_root}")
            if not raw_root.is_dir():
                raise ValueError(f"--raw-root is not a directory: {raw_root}")

            _raw_train = raw_root / "train"
            _raw_test = raw_root / "test"
            _has_split_subdirs = _raw_train.is_dir() or _raw_test.is_dir()

            if _has_split_subdirs:
                simulation_dirs = sorted(
                    path
                    for subdir in [_raw_train, _raw_test]
                    if subdir.is_dir()
                    for path in subdir.iterdir()
                    if path.is_dir() and path.name.lower() != "v_2"
                )
            else:
                simulation_dirs = sorted(
                    path for path in raw_root.iterdir()
                    if path.is_dir() and path.name.lower() != "v_2"
                )
            if not simulation_dirs:
                raise ValueError(f"No simulation folders found in: {raw_root}")

            processed_root = Path(args.processed_dir)
            processed_root.mkdir(parents=True, exist_ok=True)

            split_map = _load_train_test_split_csv(Path(args.split_csv)) if args.split_csv else {}

            # Derive split from directory structure when raw_root has train/test subdirs.
            # Entries already in split_map (from --split-csv) take precedence.
            if _has_split_subdirs:
                for _subdir, _split_name in [(_raw_train, "train"), (_raw_test, "test")]:
                    if _subdir.is_dir():
                        for _path in _subdir.iterdir():
                            if _path.is_dir() and _path.name.lower() != "v_2" and _path.name not in split_map:
                                split_map[_path.name] = _split_name

            use_split_routing = bool(split_map)
            if use_split_routing:
                (processed_root / "train").mkdir(parents=True, exist_ok=True)
                (processed_root / "test").mkdir(parents=True, exist_ok=True)

            report_path = DEFAULT_BATCH_REPORT_PATH
            report_path.parent.mkdir(parents=True, exist_ok=True)

            shared_permeability_path = Path(args.shared_permeability_path) if args.shared_permeability_path else None
            shared_porosity_path = Path(args.shared_porosity_path) if args.shared_porosity_path else None
            for label, shared_path in (
                ("shared-permeability-path", shared_permeability_path),
                ("shared-porosity-path", shared_porosity_path),
            ):
                if shared_path is None:
                    continue
                if not shared_path.exists():
                    raise ValueError(f"--{label} does not exist: {shared_path}")
                if not shared_path.is_file():
                    raise ValueError(f"--{label} is not a file: {shared_path}")

            if args.compute_global_normalization_only and not (
                use_split_routing and args.normalization_scope == "split"
            ):
                raise ValueError(
                    "--compute-global-normalization-only requires --split-csv and "
                    "--normalization-scope split in --all-simulations mode."
                )

            global_stats_by_split: dict[str, dict[str, dict[str, float]]] = {}
            if args.normalization_scope == "split" and use_split_routing:
                global_norm_dir = DEFAULT_GLOBAL_NORMALIZATION_DIR
                global_norm_dir.mkdir(parents=True, exist_ok=True)

                if args.global_normalization_stats is not None:
                    stats_path = Path(args.global_normalization_stats)
                    if not stats_path.exists():
                        raise ValueError(f"--global-normalization-stats does not exist: {stats_path}")
                    if not stats_path.is_file():
                        raise ValueError(f"--global-normalization-stats is not a file: {stats_path}")
                    stats = json.loads(stats_path.read_text(encoding="utf-8"))
                    if not isinstance(stats, dict):
                        raise ValueError(f"Invalid global normalization JSON (expected object): {stats_path}")
                    global_stats_by_split["train"] = stats
                elif args.normalize or args.compute_global_normalization_only:
                    train_dirs = [d for d in simulation_dirs if split_map.get(d.name) == "train"]
                    if not train_dirs:
                        raise ValueError(
                            "No split=train simulations found. Cannot compute global normalization stats."
                        )
                    stats = _compute_global_minmax_for_simulations(
                        train_dirs,
                        nz=args.nz, nj=args.nj, ni=args.ni,
                        injection_sheet=args.injection_sheet,
                    )
                    global_stats_by_split["train"] = stats
                    save_global_stats(stats, DEFAULT_GLOBAL_STATS_FILE)

                if args.compute_global_normalization_only:
                    return 0

            if args.normalize and args.normalization_scope == "split" and use_split_routing:
                if "train" not in global_stats_by_split:
                    raise ValueError(
                        "Global normalization stats for split=train are missing. "
                        "Provide --global-normalization-stats or run with "
                        "--compute-global-normalization-only first."
                    )

            # ----------------------------------------------------------------
            # PARALLEL BRANCH
            # ----------------------------------------------------------------
            if args.parallel:
                from .pipeline.parallel import run_batch_pipeline  # noqa: PLC0415

                retry_failed: list[str] | None = None
                if args.retry_failed:
                    retry_report_path = Path(args.retry_failed)
                    if retry_report_path.exists():
                        prev = json.loads(retry_report_path.read_text(encoding="utf-8"))
                        retry_failed = [
                            r["sim_name"]
                            for r in prev.get("worker_results", [])
                            if r.get("status") == "error"
                        ]
                        print(f"[RETRY] Retrying {len(retry_failed)} failed simulations from {retry_report_path}")

                if args.normalization_scope == "split" and use_split_routing:
                    par_stats_path: Path | None = DEFAULT_GLOBAL_STATS_FILE
                    if "train" in global_stats_by_split:
                        save_global_stats(global_stats_by_split["train"], par_stats_path)
                else:
                    par_stats_path = None

                par_report = run_batch_pipeline(
                    sim_dirs=simulation_dirs,
                    grid=GridShape(nz=args.nz, nj=args.nj, ni=args.ni),
                    output_dir=processed_root,
                    n_workers=args.n_workers,
                    normalize=args.normalize,
                    split_map=split_map if use_split_routing else None,
                    stats_path=par_stats_path,
                    fmt="pt" if args.torch_output else "npz",
                    skip_existing=args.skip_existing_outputs,
                    retry_failed=retry_failed,
                    injection_sheet=args.injection_sheet,
                    full_tensor_output=args.full_tensor_output,
                )

                par_report_path = report_path.parent / "batch_parallel_report.json"
                par_report_path.write_text(json.dumps(par_report.to_dict(), indent=2), encoding="utf-8")
                _print_execution_time("total", round(perf_counter() - start_total, 6))
                print(json.dumps(par_report.to_dict(), indent=2))
                return 0 if par_report.failed == 0 else 1

            # ----------------------------------------------------------------
            # SERIAL BATCH LOOP
            # ----------------------------------------------------------------
            simulations_report: dict[str, Any] = {}
            skipped_simulations: list[str] = []
            failed_simulations: list[str] = []
            succeeded_simulations: list[str] = []
            existing_outputs_simulations: list[str] = []

            for sim_dir in simulation_dirs:
                sim_name = sim_dir.name
                split = split_map.get(sim_name) if use_split_routing else None
                if use_split_routing:
                    if split not in ("train", "test"):
                        skipped_simulations.append(sim_name)
                        print(f"[SKIP] {sim_name}: not present in split CSV ({args.split_csv})")
                        simulations_report[sim_name] = {
                            "input_dir": str(sim_dir),
                            "processed_dir": None,
                            "skipped": True,
                            "reason": "missing_in_split_csv",
                        }
                        continue
                    sim_processed_dir = processed_root / split / sim_name
                else:
                    sim_processed_dir = processed_root / sim_name
                sim_processed_dir.mkdir(parents=True, exist_ok=True)

                sfvd_report_path = sim_processed_dir / "layer_cubes_report.json"
                if args.skip_existing_outputs and sfvd_report_path.exists():
                    print(f"[SKIP] {sim_name}: existing outputs detected ({sfvd_report_path})")
                    existing_outputs_simulations.append(sim_name)
                    simulations_report[sim_name] = {
                        "input_dir": str(sim_dir),
                        "processed_dir": str(sim_processed_dir),
                        "existing_outputs": True,
                        "reason": "skip_existing_outputs",
                    }
                    _write_batch_report_incremental(
                        report_path=report_path, raw_root=raw_root, processed_root=processed_root,
                        simulation_dirs=simulation_dirs, simulations_report=simulations_report,
                        succeeded_simulations=succeeded_simulations,
                        existing_outputs_simulations=existing_outputs_simulations,
                        skipped_simulations=skipped_simulations, failed_simulations=failed_simulations,
                        start_total=start_total,
                    )
                    continue

                try:
                    discovered = discover_simulation_inputs(sim_dir)
                    print(f"[SIM] {sim_name}: input={sim_dir} -> output={sim_processed_dir}")
                    normalize_this_sim = bool(args.normalize)
                    if use_split_routing and split == "test":
                        normalize_this_sim = False

                    global_stats = None
                    if normalize_this_sim and args.normalization_scope == "split" and use_split_routing:
                        global_stats = global_stats_by_split.get("train", None)

                    sim_reports = _run_requested_pipelines(
                        sf_path=discovered["sf_path"],
                        vd_path=discovered["vd_path"],
                        permeability_path=discovered["permeability_path"] or shared_permeability_path,
                        porosity_path=discovered["porosity_path"] or shared_porosity_path,
                        cohesion_path=discovered["cohesion_path"],
                        afi_path=discovered["afi_path"],
                        pressure_path=discovered["pressure_path"],
                        gas_saturation_path=discovered["gas_saturation_path"],
                        injection_path=discovered["injection_path"],
                        injection_sheet=args.injection_sheet,
                        processed_dir=sim_processed_dir,
                        nz=args.nz, nj=args.nj, ni=args.ni,
                        normalize=normalize_this_sim,
                        full_tensor_output=args.full_tensor_output,
                        torch_output=args.torch_output,
                        global_stats=global_stats,
                        skip_existing_pipelines=args.skip_existing_pipelines,
                    )
                    simulations_report[sim_name] = {
                        "input_dir": str(sim_dir),
                        "processed_dir": str(sim_processed_dir),
                        "split": split,
                        "reports": sim_reports,
                    }
                    succeeded_simulations.append(sim_name)

                except ValueError as exc:
                    msg = str(exc).lower()
                    if args.skip_missing_required and ("missing required files" in msg or "has no files" in msg):
                        skipped_simulations.append(sim_name)
                        print(f"[SKIP] {sim_name}: {exc}")
                        simulations_report[sim_name] = {
                            "input_dir": str(sim_dir),
                            "processed_dir": str(sim_processed_dir),
                            "skipped": True, "error": str(exc),
                        }
                        _write_batch_report_incremental(
                            report_path=report_path, raw_root=raw_root, processed_root=processed_root,
                            simulation_dirs=simulation_dirs, simulations_report=simulations_report,
                            succeeded_simulations=succeeded_simulations,
                            existing_outputs_simulations=existing_outputs_simulations,
                            skipped_simulations=skipped_simulations, failed_simulations=failed_simulations,
                            start_total=start_total,
                        )
                        continue
                    failed_simulations.append(sim_name)
                    raise

                except Exception as exc:
                    if args.skip_missing_required:
                        failed_simulations.append(sim_name)
                        print(f"[FAIL] {sim_name}: {exc}")
                        simulations_report[sim_name] = {
                            "input_dir": str(sim_dir),
                            "processed_dir": str(sim_processed_dir),
                            "failed": True, "error": str(exc),
                        }
                        _write_batch_report_incremental(
                            report_path=report_path, raw_root=raw_root, processed_root=processed_root,
                            simulation_dirs=simulation_dirs, simulations_report=simulations_report,
                            succeeded_simulations=succeeded_simulations,
                            existing_outputs_simulations=existing_outputs_simulations,
                            skipped_simulations=skipped_simulations, failed_simulations=failed_simulations,
                            start_total=start_total,
                        )
                        continue
                    raise

                _write_batch_report_incremental(
                    report_path=report_path, raw_root=raw_root, processed_root=processed_root,
                    simulation_dirs=simulation_dirs, simulations_report=simulations_report,
                    succeeded_simulations=succeeded_simulations,
                    existing_outputs_simulations=existing_outputs_simulations,
                    skipped_simulations=skipped_simulations, failed_simulations=failed_simulations,
                    start_total=start_total,
                )

            total_seconds = round(perf_counter() - start_total, 6)
            batch_report = {
                "mode": "multi-simulation",
                "raw_root": str(raw_root),
                "processed_root": str(processed_root),
                "simulations_count": len(simulation_dirs),
                "simulations_succeeded": succeeded_simulations,
                "simulations_existing_outputs": existing_outputs_simulations,
                "simulations_skipped": skipped_simulations,
                "simulations_failed": failed_simulations,
                "simulations": simulations_report,
                "execution_seconds_total": total_seconds,
            }
            report_path.write_text(json.dumps(batch_report, indent=2), encoding="utf-8")
            _print_execution_time("total", total_seconds)
            print(json.dumps(batch_report, indent=2))
            if succeeded_simulations or existing_outputs_simulations:
                return 0
            return 1

        # ----------------------------------------------------------------
        # SINGLE-SIMULATION MODE
        # ----------------------------------------------------------------
        reports = _run_requested_pipelines(
            sf_path=args.sf_path,
            vd_path=args.vd_path,
            permeability_path=args.permeability_path,
            porosity_path=args.porosity_path,
            cohesion_path=args.cohesion_path,
            afi_path=args.afi_path,
            pressure_path=args.pressure_path,
            gas_saturation_path=args.gas_saturation_path,
            injection_path=args.injection_path,
            injection_sheet=args.injection_sheet,
            processed_dir=args.processed_dir,
            nz=args.nz, nj=args.nj, ni=args.ni,
            normalize=args.normalize,
            full_tensor_output=args.full_tensor_output,
            torch_output=args.torch_output,
            global_stats=None,
            skip_existing_pipelines=args.skip_existing_pipelines,
        )

        total_seconds = round(perf_counter() - start_total, 6)
        reports["execution_seconds_total"] = total_seconds
        _print_execution_time("total", total_seconds)
        print(json.dumps(reports, indent=2))
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
