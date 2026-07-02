"""
normalize_train_test.py

POST-processing min-max normalization for cmg2tensor outputs.

Operates on already-generated .pt files. Reads the split assignment from
reports/train_test_split_all.csv (produced by make_split_v2.py).

Strategy:
  - Stats (min/max) computed ONLY from the train split
  - Same stats applied to both train and test (no data leakage)
  - TARGET variables (SF, VD) are NEVER normalized — copied as-is
  - Input features (AFI, COHESION, PERMEABILITY, POROSITY, injection) are normalized
  - Output: data/processed/train_test_norm/{train,test}/
  - Metadata: data/processed/train_test_norm/normalization_metadata.json

Incremental stats cache (reports/sim_stats_cache.json):
  - Stores per-simulation min/max so new sims can be added without
    re-reading all existing ones. Only new/unseen simulations are scanned.
  - Use --rebuild-cache to force a full re-scan.

Usage (run from project root):
  python scripts/normalize_train_test.py
  python scripts/normalize_train_test.py --rebuild-cache
  python scripts/normalize_train_test.py --dry-run
  python scripts/normalize_train_test.py --split-csv reports/train_test_split_all.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from modelo_itm.etl.config import DEFAULT_GLOBAL_STATS_FILE
from modelo_itm.etl.stats import load_global_stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parents[1]

# Maps subdirectory name → ordered variable names (index = channel axis 0)
CUBE_SUBDIR_VARS: dict[str, list[str]] = {
    "layer_cubes":              ["SF", "VD"],
    "afi_layer_cubes":          ["AFI"],
    "cohesion_layer_cubes":     ["COHESION"],
    "permeability_layer_cubes": ["PERMEABILITY"],
    "porosity_layer_cubes":     ["POROSITY"],
    "pressure_layer_cubes":     ["PRESSURE"],
}

# Variables that must NOT be normalized (model targets)
TARGET_VARS: frozenset[str] = frozenset({"SF", "VD"})

INJECTION_SUBDIR = "injection_name_tensors"

DEFAULT_SPLIT_CSV      = BASE_DIR / "reports" / "train_test_split_all.csv"
DEFAULT_STATS_PATH     = DEFAULT_GLOBAL_STATS_FILE
DEFAULT_OUT_DIR        = BASE_DIR / "data" / "processed" / "train_test_norm"
DEFAULT_SIM_CACHE_PATH = BASE_DIR / "reports" / "sim_stats_cache.json"
PROCESSED_DIR          = BASE_DIR / "data" / "processed"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class VarStats(NamedTuple):
    min: float
    max: float


Stats = dict[str, VarStats]  # {var_name: VarStats}


# ---------------------------------------------------------------------------
# Split CSV discovery
# ---------------------------------------------------------------------------

def _resolve_sim_path(row: dict, base_dir: Path) -> Path | None:
    """
    Resolve the simulation directory from a CSV row.
    Tries in order:
      1. data/processed/{split}/{simulation_name}  (post-split structure)
      2. source_dir from CSV                       (legacy / flat structure)
    """
    split = row.get("split", "")
    sim_name = row.get("simulation_name", "")
    if split and sim_name:
        candidate = base_dir / "data" / "processed" / split / sim_name
        if candidate.is_dir():
            return candidate
    legacy = base_dir / row["source_dir"]
    if legacy.is_dir():
        return legacy
    return None


def discover_split_dirs(
    split_csv: Path,
    base_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """
    Read split CSV and return (train_dirs, test_dirs) as absolute Paths.
    Rows with status='raw' are skipped (no .pt files to normalize yet).
    """
    train_dirs: list[Path] = []
    test_dirs: list[Path] = []

    with open(split_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "raw":
                continue
            sim_path = _resolve_sim_path(row, base_dir)
            if sim_path is None:
                print(f"  [WARN] Not found, skipping: {row.get('simulation_name')}", file=sys.stderr)
                continue
            if row["split"] == "train":
                train_dirs.append(sim_path)
            else:
                test_dirs.append(sim_path)

    return train_dirs, test_dirs


# ---------------------------------------------------------------------------
# Stats: load or compute from train
# ---------------------------------------------------------------------------

def _load_pt(path: Path) -> dict:
    payload = torch.load(path, weights_only=True)
    if isinstance(payload, torch.Tensor):
        return {"cube": payload}
    return payload


def _tensor_key(payload: dict) -> str:
    if "cube" in payload:
        return "cube"
    if "tensor" in payload:
        return "tensor"
    raise KeyError(f"No known tensor key. Keys: {list(payload.keys())}")


def _tensor_numpy(payload: dict) -> np.ndarray:
    t = payload[_tensor_key(payload)]
    return t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def load_stats_from_json(path: Path) -> Stats:
    raw = load_global_stats(path)
    return {var: VarStats(min=float(v["min"]), max=float(v["max"])) for var, v in raw.items()}


# ---------------------------------------------------------------------------
# Per-simulation stats cache
# ---------------------------------------------------------------------------

SimCache = dict[str, dict[str, dict[str, float]]]  # {sim_name: {var: {min, max}}}


def load_sim_cache(path: Path) -> SimCache:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_sim_cache(cache: SimCache, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _scan_sim_stats(sim_dir: Path) -> dict[str, dict[str, float]]:
    """Compute per-variable {min, max} for a single simulation directory."""
    accum: dict[str, tuple[float, float]] = {}

    def _update(var: str, arr: np.ndarray) -> None:
        if var in TARGET_VARS:
            return
        vmin, vmax = float(arr.min()), float(arr.max())
        if var in accum:
            accum[var] = (min(accum[var][0], vmin), max(accum[var][1], vmax))
        else:
            accum[var] = (vmin, vmax)

    for subdir_name, fallback_vars in CUBE_SUBDIR_VARS.items():
        subdir = sim_dir / subdir_name
        if not subdir.exists():
            continue
        for pt_file in subdir.glob("*.pt"):
            payload = _load_pt(pt_file)
            arr = _tensor_numpy(payload)
            var_names: list[str] = payload.get("variables") or fallback_vars
            for ch, var in enumerate(var_names):
                if ch < arr.shape[0]:
                    _update(var, arr[ch])

    inj_dir = sim_dir / INJECTION_SUBDIR
    if inj_dir.exists():
        for pt_file in inj_dir.glob("*.pt"):
            payload = _load_pt(pt_file)
            var = payload.get("name") or pt_file.stem
            _update(var, _tensor_numpy(payload))

    return {var: {"min": lo, "max": hi} for var, (lo, hi) in accum.items()}


def _aggregate_cache(cache: SimCache) -> Stats:
    """Merge all per-sim stats into global min/max."""
    accum: dict[str, tuple[float, float]] = {}
    for sim_stats in cache.values():
        for var, vs in sim_stats.items():
            vmin, vmax = vs["min"], vs["max"]
            if var in accum:
                accum[var] = (min(accum[var][0], vmin), max(accum[var][1], vmax))
            else:
                accum[var] = (vmin, vmax)
    return {var: VarStats(min=lo, max=hi) for var, (lo, hi) in accum.items()}


def compute_stats_from_train(
    train_dirs: list[Path],
    cache: SimCache,
    cache_path: Path,
    rebuild: bool = False,
) -> Stats:
    """
    Compute global min/max from train simulations using an incremental cache.
    Only simulations not already in the cache are scanned; the rest are reused.
    Saves the updated cache to cache_path.
    """
    new_sims = [d for d in train_dirs if rebuild or d.name not in cache]
    cached_sims = [d for d in train_dirs if not rebuild and d.name in cache]

    print(f"  Cached : {len(cached_sims)} sims (skipping scan)")
    print(f"  New    : {len(new_sims)} sims to scan")

    for i, sim_dir in enumerate(sorted(new_sims), 1):
        print(f"    [{i:>3}/{len(new_sims)}] {sim_dir.name}", end="\r", flush=True)
        cache[sim_dir.name] = _scan_sim_stats(sim_dir)
    if new_sims:
        print()

    save_sim_cache(cache, cache_path)
    print(f"  Cache saved -> {cache_path}  ({len(cache)} sims total)")

    return _aggregate_cache({k: cache[k] for k in cache if k in {d.name for d in train_dirs}})


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def minmax_scale(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = vmax - vmin
    if span <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr.astype(np.float64) - vmin) / span).astype(np.float32)


# ---------------------------------------------------------------------------
# File-level normalization
# ---------------------------------------------------------------------------

def normalize_cube_pt(
    src: Path,
    dst: Path,
    var_names: list[str],
    stats: Stats,
) -> None:
    """
    Normalize each channel of a cube .pt file.
    Channels whose variable is in TARGET_VARS are copied without modification.
    """
    payload = _load_pt(src)
    key = _tensor_key(payload)
    arr = _tensor_numpy(payload).copy().astype(np.float32)

    effective_vars: list[str] = payload.get("variables") or var_names

    for ch, var in enumerate(effective_vars):
        if ch >= arr.shape[0]:
            break
        if var in TARGET_VARS:
            continue  # preserve targets as-is
        if var not in stats:
            continue
        s = stats[var]
        arr[ch] = minmax_scale(arr[ch], s.min, s.max)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, key: torch.from_numpy(arr)}, dst)


def normalize_injection_pt(src: Path, dst: Path, stats: Stats) -> None:
    """Normalize a 1-D injection .pt file using payload['name'] as variable key."""
    payload = _load_pt(src)
    var = payload.get("name") or src.stem
    arr = _tensor_numpy(payload).copy().astype(np.float32)

    if var in stats:
        s = stats[var]
        arr = minmax_scale(arr, s.min, s.max)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "tensor": torch.from_numpy(arr)}, dst)


# ---------------------------------------------------------------------------
# Simulation & split processing
# ---------------------------------------------------------------------------

def _patch_normalization(report: dict, stats: Stats) -> dict:
    def _filled(per_variable: dict) -> dict:
        result = {}
        for var, val in per_variable.items():
            if var in TARGET_VARS:
                result[var] = {**val, "normalized": False, "reason": "target variable"}
            elif var in stats:
                result[var] = {
                    "min": stats[var].min,
                    "max": stats[var].max,
                    "span": stats[var].max - stats[var].min,
                }
            else:
                result[var] = val
        return {
            "applied": True,
            "method": "minmax",
            "scope": "split",
            "per_variable": result,
        }

    if "normalization" in report and isinstance(report.get("normalization"), dict):
        n = report["normalization"]
        if "per_variable" in n:
            report["normalization"] = _filled(n["per_variable"])

    if "series" in report and isinstance(report.get("series"), dict):
        for well_data in report["series"].values():
            if isinstance(well_data, dict) and "normalization" in well_data:
                n = well_data["normalization"]
                if "per_variable" in n:
                    well_data["normalization"] = _filled(n["per_variable"])

    return report


def _copy_report_with_updated_paths(
    src: Path, dst: Path, sim_src: Path, sim_dst: Path, stats: Stats | None = None
) -> None:
    with open(src, encoding="utf-8") as f:
        report = json.load(f)

    abs_src = str(sim_src.resolve()).replace("\\", "/")
    abs_dst = str(sim_dst.resolve()).replace("\\", "/")

    def _rel_tail(abs_path: str) -> str:
        for i, part in enumerate(abs_path.split("/")):
            if part.lower() == "data":
                return "/".join(abs_path.split("/")[i:])
        return abs_path

    rel_src = _rel_tail(abs_src)
    rel_dst = _rel_tail(abs_dst)
    replacements = [(abs_src, abs_dst), (rel_src, rel_dst)]

    def _fix(val: str) -> str:
        normalized = val.replace("\\", "/")
        for needle, replacement in replacements:
            if needle in normalized:
                return normalized.replace(needle, replacement)
        return val

    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, str):
            return _fix(obj)
        return obj

    updated = _walk(report)
    if stats is not None:
        updated = _patch_normalization(updated, stats)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)


def process_simulation(sim_src: Path, sim_dst: Path, stats: Stats, dry_run: bool) -> None:
    if not dry_run:
        sim_dst.mkdir(parents=True, exist_ok=True)

    for item in sorted(sim_src.iterdir()):
        dst_item = sim_dst / item.name

        if item.is_file() and item.suffix == ".json":
            if not dry_run:
                _copy_report_with_updated_paths(item, dst_item, sim_src, sim_dst, stats=stats)

        elif item.is_dir() and item.name in CUBE_SUBDIR_VARS:
            var_names = CUBE_SUBDIR_VARS[item.name]
            if not dry_run:
                dst_item.mkdir(parents=True, exist_ok=True)
            for pt_file in sorted(item.glob("*.pt")):
                if not dry_run:
                    normalize_cube_pt(pt_file, dst_item / pt_file.name, var_names, stats)

        elif item.is_dir() and item.name == INJECTION_SUBDIR:
            if not dry_run:
                dst_item.mkdir(parents=True, exist_ok=True)
            for pt_file in sorted(item.glob("*.pt")):
                if not dry_run:
                    normalize_injection_pt(pt_file, dst_item / pt_file.name, stats)

        elif item.is_dir():
            if not dry_run:
                shutil.copytree(item, dst_item, dirs_exist_ok=True)


def process_dirs(
    sim_dirs: list[Path],
    out_split_dir: Path,
    stats: Stats,
    split_name: str,
    dry_run: bool,
) -> None:
    action = "[dry-run] " if dry_run else ""
    print(f"  {action}{split_name}: {len(sim_dirs)} simulations → {out_split_dir}")
    for i, sim_dir in enumerate(sorted(sim_dirs), 1):
        sim_dst = out_split_dir / sim_dir.name
        process_simulation(sim_dir, sim_dst, stats, dry_run)
        print(f"    [{i:>3}/{len(sim_dirs)}] {sim_dir.name}", end="\r", flush=True)
    print()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_metadata(stats: Stats, out_path: Path, dry_run: bool) -> None:
    payload = {
        var: {"min": s.min, "max": s.max}
        for var, s in sorted(stats.items())
    }
    payload["_targets_excluded"] = sorted(TARGET_VARS)
    if dry_run:
        print(f"  [dry-run] Would write metadata → {out_path}")
        print(json.dumps(payload, indent=2))
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Metadata saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Min-Max normalization: inputs only (SF/VD targets preserved)"
    )
    p.add_argument(
        "--split-csv", type=Path, default=DEFAULT_SPLIT_CSV,
        help=f"Split assignment CSV (default: {DEFAULT_SPLIT_CSV})",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--stats-path", type=Path, default=DEFAULT_STATS_PATH,
        help=f"Pre-computed train stats JSON (default: {DEFAULT_STATS_PATH})",
    )
    p.add_argument(
        "--recompute-stats", action="store_true",
        help="Ignore global stats JSON and recompute (still uses sim cache unless --rebuild-cache).",
    )
    p.add_argument(
        "--rebuild-cache", action="store_true",
        help="Force a full re-scan of all train sims, ignoring the sim stats cache.",
    )
    p.add_argument(
        "--sim-cache", type=Path, default=DEFAULT_SIM_CACHE_PATH,
        help=f"Per-simulation stats cache JSON (default: {DEFAULT_SIM_CACHE_PATH})",
    )
    p.add_argument(
        "--stats-only", action="store_true",
        help="Only compute and save stats/cache, skip normalization.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without reading or writing any data files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.split_csv.exists():
        print(f"ERROR: split CSV not found: {args.split_csv}", file=sys.stderr)
        print("  Run:  python scripts/make_split_v2.py", file=sys.stderr)
        sys.exit(1)

    out_dir   = args.out_dir
    train_dst = out_dir / "train"
    test_dst  = out_dir / "test"
    meta_path = out_dir / "normalization_metadata.json"

    # ── Step 1: Discover simulation directories from CSV ────────────────────
    print(f"[1/5] Reading split from {args.split_csv}")
    train_dirs, test_dirs = discover_split_dirs(args.split_csv, BASE_DIR)
    print(f"      Train: {len(train_dirs)} sims  |  Test: {len(test_dirs)} sims")
    print(f"      Targets excluded from normalization: {sorted(TARGET_VARS)}")

    # ── Step 2: Obtain train stats (inputs only) ────────────────────────────
    if not args.recompute_stats and not args.rebuild_cache and args.stats_path.exists():
        print(f"[2/5] Loading pre-computed global stats from {args.stats_path}")
        stats = load_stats_from_json(args.stats_path)
    else:
        print(f"[2/5] Computing stats from train split (incremental cache)...")
        cache = {} if args.rebuild_cache else load_sim_cache(args.sim_cache)
        stats = compute_stats_from_train(
            train_dirs,
            cache=cache,
            cache_path=args.sim_cache,
            rebuild=args.rebuild_cache,
        )

    print(f"      Variables in stats ({len(stats)}): {sorted(stats)}")

    # ── Step 3: Save normalization metadata ─────────────────────────────────
    print(f"[3/5] Saving normalization metadata...")
    save_metadata(stats, meta_path, dry_run=args.dry_run)

    if args.stats_only:
        print("\nDone (--stats-only: normalization skipped).")
        return

    # ── Step 4: Normalize train ─────────────────────────────────────────────
    print(f"[4/5] Normalizing train split...")
    process_dirs(train_dirs, train_dst, stats, "train", dry_run=args.dry_run)

    # ── Step 5: Normalize test (reuse train stats) ───────────────────────────
    print(f"[5/5] Normalizing test split (using train stats)...")
    process_dirs(test_dirs, test_dst, stats, "test", dry_run=args.dry_run)

    print(f"\nDone.  Output: {out_dir}")


if __name__ == "__main__":
    main()
