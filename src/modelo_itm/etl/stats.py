"""
Fast statistics scanner for cmg2tensor Phase 1.

Computes per-variable global min/max across simulations via line-by-line scanning,
avoiding the need to load full 3 GB files into memory.

All public functions are module-level (not closures) so they can be pickled by
ProcessPoolExecutor on Windows (spawn context).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_INJECTION_PARAMETER, DEFAULT_INCLUDE_INJECTION_NAMES
from .parse_txt import _parse_txt_with_times, _parse_time_day

_RESULTS_PROP_MINMAX_RE = re.compile(
    r"RESULTS\s+PROP\s+Minimum\s+Value:\s*"
    r"([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s+"
    r"Maximum\s+Value:\s*"
    r"([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)
_TIME_LINE_RE = re.compile(r"\*\*\s*TIME\s*=\s*([-+]?(?:\d*\.\d+|\d+))")

_DEFAULT_INCLUDE_INJECTION_NAMES = DEFAULT_INCLUDE_INJECTION_NAMES
_DEFAULT_INJECTION_PARAMETER = DEFAULT_INJECTION_PARAMETER


@dataclass
class FileScanResult:
    """Stats extracted from scanning one CMG .txt file."""

    path: Path
    var_name: str
    time_days: list[int]
    vmin: float | None          # None when no values could be found
    vmax: float | None
    minmax_source: str          # "header" | "tensor" | "none"
    n_minmax_lines: int


@dataclass
class SimulationScanResult:
    """Aggregated scan results for one simulation directory."""

    sim_name: str
    sim_dir: Path
    file_results: list[FileScanResult] = field(default_factory=list)
    error: str | None = None


def scan_file_for_stats(
    path: Path,
    var_name: str,
    *,
    nz: int,
    nj: int,
    ni: int,
    encoding: str = "utf-8",
) -> FileScanResult:
    """
    Single-pass line scan of one CMG .txt file for min/max statistics.

    Strategy:
    1. Scan line by line for TIME headers and "RESULTS PROP" min/max lines.
    2. If RESULTS PROP lines are present → return header-derived stats
       (minmax_source="header"). Extremely fast, no tensor allocation.
    3. If absent → parse the full tensor via the streaming _parse_txt_with_times,
       extract nanmin/nanmax, then delete the tensor immediately before returning
       (minmax_source="tensor"). This is the Fix #3 fallback.

    Memory: O(1) for the header path; O(NZ*NJ*NI*T*sizeof(float32)) for fallback
    (same as a normal parse, but tensor is released before the function returns).
    """
    time_days: list[int] = []
    global_min = float("inf")
    global_max = float("-inf")
    n_minmax = 0

    with path.open("r", encoding=encoding, errors="ignore") as fh:
        for line in fh:
            if "**" in line and "TIME" in line and "=" in line:
                tm = _TIME_LINE_RE.search(line)
                if tm:
                    day_int = _parse_time_day(tm.group(1), path)
                    time_days.append(day_int)
                continue

            if "RESULTS" in line and "PROP" in line and "Minimum Value" in line:
                mm = _RESULTS_PROP_MINMAX_RE.search(line)
                if mm:
                    n_minmax += 1
                    global_min = min(global_min, float(mm.group(1)))
                    global_max = max(global_max, float(mm.group(2)))

    if n_minmax > 0:
        return FileScanResult(
            path=path,
            var_name=var_name,
            time_days=time_days,
            vmin=float(global_min),
            vmax=float(global_max),
            minmax_source="header",
            n_minmax_lines=n_minmax,
        )

    # Fallback: parse full tensor, grab stats, release immediately.
    tensor_4d, full_times = _parse_txt_with_times(
        path, nz, nj, ni, dtype=np.float32, strict=False
    )
    vmin = float(np.nanmin(tensor_4d))
    vmax = float(np.nanmax(tensor_4d))
    del tensor_4d  # release RAM before returning

    return FileScanResult(
        path=path,
        var_name=var_name,
        time_days=full_times if full_times else time_days,
        vmin=vmin,
        vmax=vmax,
        minmax_source="tensor",
        n_minmax_lines=0,
    )


def scan_simulation_for_stats(
    sim_dir: Path,
    *,
    nz: int,
    nj: int,
    ni: int,
    discovered: dict[str, Any],
    injection_sheet: str = "Well Summary",
    include_injection_names: tuple[str, ...] = _DEFAULT_INCLUDE_INJECTION_NAMES,
) -> SimulationScanResult:
    """
    Scan one simulation directory for per-variable min/max statistics.

    This is the **top-level Phase 1 worker function** submitted to
    ProcessPoolExecutor. It must remain a module-level function (not a closure or
    lambda) to be picklable on Windows (spawn context).

    Parameters
    ----------
    sim_dir:
        Path to the simulation input directory.
    discovered:
        Dict pre-computed by _discover_simulation_inputs(sim_dir) in app.py.
        Keys: sf_path, vd_path, permeability_path, porosity_path, cohesion_path,
              afi_path, pressure_path, gas_saturation_path, injection_path.
        Values are Path objects or None.
    """
    result = SimulationScanResult(sim_name=sim_dir.name, sim_dir=sim_dir)
    try:
        variable_paths: list[tuple[str, Path | None]] = [
            ("SF", discovered.get("sf_path")),
            ("VD", discovered.get("vd_path")),
            ("PERMEABILITY", discovered.get("permeability_path")),
            ("POROSITY", discovered.get("porosity_path")),
            ("COHESION", discovered.get("cohesion_path")),
            ("AFI", discovered.get("afi_path")),
            ("PRESSURE", discovered.get("pressure_path")),
            ("GAS_SATURATION", discovered.get("gas_saturation_path")),
        ]

        sf_times: list[int] = []
        for var_name, var_path in variable_paths:
            if var_path is None:
                continue
            fr = scan_file_for_stats(Path(var_path), var_name, nz=nz, nj=nj, ni=ni)
            result.file_results.append(fr)
            if var_name == "SF":
                sf_times = fr.time_days

        # Injection series (optional, requires pandas + openpyxl)
        inj_path = discovered.get("injection_path")
        if inj_path is not None and sf_times:
            inj_results = _scan_injection_for_stats(
                Path(inj_path),
                reference_time_days=sf_times,
                sheet_name=injection_sheet,
                include_names=include_injection_names,
            )
            result.file_results.extend(inj_results)

    except Exception as exc:  # noqa: BLE001
        import traceback as _tb
        result.error = f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"

    return result


def _scan_injection_for_stats(
    path: Path,
    *,
    reference_time_days: list[int],
    sheet_name: str,
    include_names: tuple[str, ...],
) -> list[FileScanResult]:
    """Extract min/max for each injection series from an Excel file."""
    try:
        import pandas as pd
    except ImportError:
        return []

    df = pd.read_excel(path, sheet_name=sheet_name)
    if not {"Time (day)", "Name", "Value"}.issubset(df.columns):
        return []

    include_set = {n.strip().upper() for n in include_names}
    work = df[["Time (day)", "Name", "Value"]].copy()
    work["name_norm"] = work["Name"].astype(str).str.strip().str.upper()
    work["day"] = pd.to_numeric(work["Time (day)"], errors="coerce").round().astype("Int64")
    work["value"] = pd.to_numeric(work["Value"], errors="coerce")
    work = work.dropna(subset=["name_norm", "day", "value"])
    work["day"] = work["day"].astype(int)
    work = work[work["name_norm"].isin(include_set)]

    if "Parameter" in df.columns:
        param_norm = df["Parameter"].astype(str).str.strip().str.upper()
        work = work[param_norm.reindex(work.index).eq(_DEFAULT_INJECTION_PARAMETER.strip().upper())]

    results: list[FileScanResult] = []
    for name_upper, grp in work.groupby("name_norm"):
        vals = grp["value"].to_numpy(dtype=np.float32)
        if vals.size == 0:
            continue
        original_name = grp["Name"].astype(str).str.strip().iloc[0]
        results.append(
            FileScanResult(
                path=path,
                var_name=original_name,
                time_days=reference_time_days,
                vmin=float(np.nanmin(vals)),
                vmax=float(np.nanmax(vals)),
                minmax_source="tensor",
                n_minmax_lines=0,
            )
        )
    return results


def merge_stats(
    global_stats: dict[str, dict[str, float]],
    sim_result: SimulationScanResult,
) -> dict[str, dict[str, float]]:
    """
    Merge per-simulation stats from a SimulationScanResult into global_stats dict.

    Silently skips simulations that errored (error is already logged in the result).
    Skips file results where vmin/vmax are None.
    """
    if sim_result.error:
        return global_stats

    for fr in sim_result.file_results:
        if fr.vmin is None or fr.vmax is None:
            continue
        var = fr.var_name
        if var not in global_stats:
            global_stats[var] = {"min": fr.vmin, "max": fr.vmax}
        else:
            global_stats[var]["min"] = min(global_stats[var]["min"], fr.vmin)
            global_stats[var]["max"] = max(global_stats[var]["max"], fr.vmax)

    return global_stats


def finalize_stats(global_stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Add 'span' field to each variable entry (matches train.json format)."""
    for var_stats in global_stats.values():
        var_stats["span"] = float(var_stats["max"] - var_stats["min"])
    return global_stats


def save_global_stats(stats: dict[str, dict[str, float]], path: Path) -> None:
    """Save global stats to JSON (compatible with current train.json format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_global_stats(path: Path) -> dict[str, dict[str, float]]:
    """Load global stats from JSON (train.json or stats_path)."""
    return json.loads(path.read_text(encoding="utf-8"))
