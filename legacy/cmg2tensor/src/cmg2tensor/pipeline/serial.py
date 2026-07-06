"""
Per-simulation pipeline functions for cmg2tensor.

This module owns everything needed to transform one simulation's raw files
into normalized tensors:

  - Utility helpers (time conversion, output clearing, normalization routing)
  - I/O: save layer cubes (.pt / .npz) and full tensors
  - run_layer_cubes_pipeline          — SF + VD → per-layer cubes
  - run_single_variable_layer_cubes_pipeline — single variable → per-layer cubes
  - run_injection_excel_pipeline      — Excel injection → aligned + normalized series
  - _run_requested_pipelines          — orchestrates the above for one simulation
  - _extract_injection_aligned_series — low-level reader (shared base for Fase 4)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..config import DAYS_PER_MONTH, DEFAULT_INJECTION_PARAMETER
from ..normalize import (
    normalize_cubes_minmax,
    normalize_cubes_minmax_with_global_stats,
    normalize_series_minmax,
    normalize_series_minmax_with_global_stats,
)
from ..parse_txt import build_layer_cubes, build_single_variable_layer_cubes

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

REPORT_TIME_IDS_PREVIEW_LIMIT = 20


def _print_cube_dimensions(report: dict[str, Any], label: str) -> None:
    cube_shape = report.get("cube_shape")
    num_layers = report.get("num_layers")
    print(f"[DIM] {label}: {num_layers} capas, cada capa {cube_shape} (V, T, NJ, NI)")


def _print_execution_time(label: str, seconds: float) -> None:
    print(f"[TIME] {label}: {seconds:.3f}s")


def _clear_existing_outputs(directory: Path) -> None:
    for stale in directory.glob("layer_cube_k*.npz"):
        stale.unlink()
    for stale in directory.glob("layer_cube_k*.pt"):
        stale.unlink()


def _days_to_month_ids(time_ids_days: list[int]) -> list[int]:
    if not time_ids_days:
        return []
    base_day = time_ids_days[0]
    return [int(round((day - base_day) / DAYS_PER_MONTH)) for day in time_ids_days]


def _no_normalization_metadata(variable_names: list[str]) -> dict[str, Any]:
    return {
        "applied": False,
        "method": "none",
        "per_variable": {name: None for name in variable_names},
    }


def _slugify_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")


def _preview(values: list[int], *, limit: int) -> list[int]:
    return values if len(values) <= limit else values[:limit]


def _normalize_or_cast_cubes(
    cubes: list[np.ndarray],
    variable_order: list[str],
    normalize: bool,
    *,
    global_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if normalize:
        if global_stats is not None:
            return normalize_cubes_minmax_with_global_stats(cubes, variable_order, global_stats=global_stats)
        return normalize_cubes_minmax(cubes, variable_order)
    casted = [cube.astype(np.float32, copy=False) for cube in cubes]
    return casted, _no_normalization_metadata(variable_order)


# ---------------------------------------------------------------------------
# Timeline / report I/O helpers
# ---------------------------------------------------------------------------

def _load_reference_time_days_from_report(processed_dir: str | Path) -> list[int]:
    report_path = Path(processed_dir) / "layer_cubes_report.json"
    if not report_path.exists():
        raise ValueError(
            f"Reference report not found: {report_path}. "
            "Run SF/VD first or provide --sf-path and --vd-path in this run."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    timeline_path = report.get("timeline_path")
    if timeline_path:
        timeline_file = Path(timeline_path)
        if not timeline_file.is_absolute():
            timeline_file = Path(processed_dir) / timeline_file
        if timeline_file.exists():
            timeline = json.loads(timeline_file.read_text(encoding="utf-8"))
            if "time_ids_days" in timeline and timeline["time_ids_days"]:
                return [int(v) for v in timeline["time_ids_days"]]
    if "time_ids_days" in report and report["time_ids_days"]:
        return [int(v) for v in report["time_ids_days"]]
    if "time_ids" in report and report["time_ids"]:
        return [int(v) for v in report["time_ids"]]
    raise ValueError(f"Reference report has no time_ids: {report_path}")


def _write_timeline_file(
    processed_path: Path,
    *,
    time_ids: list[int],
    time_ids_days: list[int],
    filename: str = "timeline.json",
) -> Path:
    timeline_path = processed_path / filename
    timeline_path.write_text(
        json.dumps(
            {
                "time_ids": time_ids,
                "time_ids_days": time_ids_days,
                "time_unit": "months",
                "time_ids_count": len(time_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return timeline_path


# ---------------------------------------------------------------------------
# Tensor save helpers
# ---------------------------------------------------------------------------

def _save_layer_cubes(
    cubes: list[Any],
    time_ids: list[int],
    time_ids_days: list[int],
    variable_order: list[str],
    output_dir: Path,
    *,
    normalization: dict[str, Any],
    torch_output: bool,
) -> list[str]:
    if torch_output:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is required when --torch-output is enabled. Install `torch` first."
            ) from exc
        output_files: list[str] = []
        for idx, cube in enumerate(cubes, start=1):
            out_file = output_dir / f"layer_cube_k{idx:03d}.pt"
            cube_tensor = cube if isinstance(cube, torch.Tensor) else torch.tensor(cube, dtype=torch.float32)
            torch.save(
                {
                    "cube": cube_tensor,
                    "time_ids": time_ids,
                    "time_ids_days": time_ids_days,
                    "time_unit": "months",
                    "variables": variable_order,
                    "layer_k": idx,
                    "normalization": normalization,
                },
                out_file,
            )
            output_files.append(str(out_file))
        return output_files

    output_files = []
    for idx, cube in enumerate(cubes, start=1):
        out_file = output_dir / f"layer_cube_k{idx:03d}.npz"
        np.savez_compressed(
            out_file,
            cube=cube,
            time_ids=np.array(time_ids, dtype=np.int64),
            time_ids_days=np.array(time_ids_days, dtype=np.int64),
            time_unit=np.array(["months"]),
            variables=np.array(variable_order),
            layer_k=np.array([idx], dtype=np.int64),
            normalization_method=np.array([normalization.get("method", "none")]),
            normalization_applied=np.array([int(bool(normalization.get("applied", False)))], dtype=np.int64),
        )
        output_files.append(str(out_file))
    return output_files


def _build_full_tensor_from_layer_cubes(cubes: list[np.ndarray]) -> np.ndarray:
    """Convert list of layer cubes (V, T, J, I) into full tensor (V, T, Z, J, I)."""
    if not cubes:
        raise ValueError("Cannot build full tensor from empty cubes list.")
    return np.stack(cubes, axis=2).astype(np.float32, copy=False)


def _save_full_tensor(
    tensor_vtzji: np.ndarray,
    output_path: Path,
    *,
    time_ids: list[int],
    time_ids_days: list[int],
    variable_order: list[str],
    normalization: dict[str, Any],
    torch_output: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if torch_output:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is required when --torch-output is enabled. Install `torch` first."
            ) from exc
        torch.save(
            {
                "tensor": torch.tensor(tensor_vtzji, dtype=torch.float32),
                "time_ids": time_ids,
                "time_ids_days": time_ids_days,
                "time_unit": "months",
                "variables": variable_order,
                "axes": ["V", "T", "Z", "J", "I"],
                "normalization": normalization,
            },
            output_path,
        )
    else:
        np.savez_compressed(
            output_path,
            tensor=tensor_vtzji,
            time_ids=np.array(time_ids, dtype=np.int64),
            time_ids_days=np.array(time_ids_days, dtype=np.int64),
            time_unit=np.array(["months"]),
            variables=np.array(variable_order),
            axes=np.array(["V", "T", "Z", "J", "I"]),
            normalization_method=np.array([normalization.get("method", "none")]),
            normalization_applied=np.array([int(bool(normalization.get("applied", False)))], dtype=np.int64),
        )
    return output_path


def _save_layer_or_full_outputs(
    *,
    cubes: list[np.ndarray],
    time_ids: list[int],
    time_ids_days: list[int],
    variable_order: list[str],
    normalization: dict[str, Any],
    layers_dir: Path,
    processed_path: Path,
    full_tensor_output: bool,
    torch_output: bool,
    full_tensor_stem: str,
) -> tuple[list[str], bool, Path | None, tuple[int, ...] | None]:
    if not full_tensor_output:
        output_files = _save_layer_cubes(
            cubes, time_ids, time_ids_days, variable_order, layers_dir,
            normalization=normalization, torch_output=torch_output,
        )
        return output_files, True, None, None

    full_tensor = _build_full_tensor_from_layer_cubes(cubes)
    suffix = ".pt" if torch_output else ".npz"
    full_tensor_path = _save_full_tensor(
        full_tensor,
        processed_path / "full_tensors" / f"{full_tensor_stem}{suffix}",
        time_ids=time_ids,
        time_ids_days=time_ids_days,
        variable_order=variable_order,
        normalization=normalization,
        torch_output=torch_output,
    )
    return [], False, full_tensor_path, tuple(full_tensor.shape)


# ---------------------------------------------------------------------------
# Injection: shared reader (single source of truth for Excel reading logic)
# ---------------------------------------------------------------------------

@dataclass
class _InjectionReadResult:
    """Result of reading and aligning one injection Excel file."""
    series: dict[str, np.ndarray]                # {name: aligned float32 array}
    reference_days: list[int]
    reference_months: list[int]
    dropped_months: dict[str, list[int]]          # {name: list of dropped month ids}
    has_parameter_column: bool
    selected_parameter: str | None
    available_parameters_preview: list[str] = field(default_factory=list)
    duplicate_rows: int = 0


def _read_and_align_injection(
    input_path: Path,
    *,
    reference_time_days: list[int],
    sheet_name: str,
    include_names: tuple[str, ...],
) -> _InjectionReadResult:
    """
    Single source of truth for reading and aligning an injection Excel file.

    Reads the Excel, filters by name + parameter, converts day→month_id,
    aligns to reference_time_days, and returns raw float32 arrays per name.

    Does NOT normalize or write files — callers handle that.
    """
    import pandas as pd

    include_name_set = {name.strip().upper() for name in include_names}
    reference_days = [int(v) for v in reference_time_days]
    reference_base_day = reference_days[0]
    reference_months = _days_to_month_ids(reference_days)
    reference_month_set = set(reference_months)

    df = pd.read_excel(input_path, sheet_name=sheet_name)
    missing_columns = {"Time (day)", "Name", "Value"}.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in Excel ({sheet_name}): {sorted(missing_columns)}"
        )

    work_columns = ["Time (day)", "Name", "Value"]
    has_parameter_column = "Parameter" in df.columns
    if has_parameter_column:
        work_columns.append("Parameter")
    work = df.loc[:, work_columns].copy()
    work["name_norm"] = work["Name"].astype(str).str.strip().str.upper()
    work["day"] = pd.to_numeric(work["Time (day)"], errors="coerce").round().astype("Int64")
    work["value"] = pd.to_numeric(work["Value"], errors="coerce")
    work = work.dropna(subset=["name_norm", "day", "value"])
    work["day"] = work["day"].astype(int)
    work = work[work["name_norm"].isin(include_name_set)].copy()

    selected_parameter: str | None = None
    available_parameters_preview: list[str] = []
    if has_parameter_column:
        work["parameter_norm"] = work["Parameter"].astype(str).str.strip().str.upper()
        available_parameters = sorted(
            {p for p in work["Parameter"].dropna().astype(str).map(str.strip).unique() if p}
        )
        available_parameters_preview = available_parameters[:10]
        default_norm = DEFAULT_INJECTION_PARAMETER.strip().upper()
        work = work[work["parameter_norm"] == default_norm].copy()
        if work.empty:
            raise ValueError(
                f"Excel contains 'Parameter' column but no rows matched the required value "
                f"'{DEFAULT_INJECTION_PARAMETER}' for names {sorted(include_name_set)} in {input_path} "
                f"(sheet={sheet_name}). Available Parameter values (preview): {available_parameters_preview}"
            )
        selected_parameter = DEFAULT_INJECTION_PARAMETER

    if work.empty:
        raise ValueError(
            f"No rows found for names {sorted(include_name_set)} in {input_path} (sheet={sheet_name})."
        )

    work["month_id"] = ((work["day"] - reference_base_day) / DAYS_PER_MONTH).round().astype(int)

    duplicate_rows = int(work.duplicated(subset=["name_norm", "month_id"]).sum())
    if duplicate_rows > 0:
        work = work.groupby(["name_norm", "month_id"], as_index=False, sort=True)["value"].mean()

    series: dict[str, np.ndarray] = {}
    dropped_months: dict[str, list[int]] = {}
    for name in sorted(include_name_set):
        name_df = work[work["name_norm"] == name].copy()
        if name_df.empty:
            raise ValueError(f"Name '{name}' was requested but not found in {input_path}.")
        values_by_month = dict(zip(name_df["month_id"].astype(int), name_df["value"].astype(float)))
        missing = [month for month in reference_months if month not in values_by_month]
        if missing:
            raise ValueError(
                f"Name '{name}' is missing {len(missing)} reference timesteps. "
                f"First missing month: {missing[0]}"
            )
        dropped_months[name] = sorted(m for m in values_by_month if m not in reference_month_set)
        series[name] = np.array([values_by_month[month] for month in reference_months], dtype=np.float32)

    return _InjectionReadResult(
        series=series,
        reference_days=reference_days,
        reference_months=reference_months,
        dropped_months=dropped_months,
        has_parameter_column=has_parameter_column,
        selected_parameter=selected_parameter,
        available_parameters_preview=available_parameters_preview,
        duplicate_rows=duplicate_rows,
    )


def _extract_injection_aligned_series(
    input_path: Path,
    *,
    reference_time_days: list[int],
    sheet_name: str,
    include_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """
    Read an Excel injection file and return aligned raw arrays per well name.

    Returns {name: np.ndarray(float32)} aligned to reference_time_days.
    Does NOT normalize or save — callers decide what to do with the arrays.
    """
    result = _read_and_align_injection(
        input_path,
        reference_time_days=reference_time_days,
        sheet_name=sheet_name,
        include_names=include_names,
    )
    return result.series


# ---------------------------------------------------------------------------
# Public pipeline functions
# ---------------------------------------------------------------------------

def run_injection_excel_pipeline(
    input_path: str | Path,
    *,
    reference_time_days: list[int],
    sheet_name: str = "Well Summary",
    include_names: tuple[str, ...] = ("TENE-1", "TENE-2"),
    processed_dir: str | Path = "data/processed",
    normalize: bool = True,
    torch_output: bool = True,
    global_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Read, align, normalize, and save injection series from an Excel file."""
    start = perf_counter()
    if not reference_time_days:
        raise ValueError("reference_time_days is empty. Cannot align injection series.")

    input_file = Path(input_path)
    processed_path = Path(processed_dir)
    output_dir = processed_path / "injection_name_tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("injection_*.pt"):
        stale.unlink()
    for stale in output_dir.glob("injection_*.npz"):
        stale.unlink()

    # --- Read and align (shared logic) ---
    try:
        read = _read_and_align_injection(
            input_file,
            reference_time_days=reference_time_days,
            sheet_name=sheet_name,
            include_names=include_names,
        )
    except ImportError as exc:
        raise RuntimeError(
            "pandas/openpyxl are required to process Excel input. Install them in your environment."
        ) from exc

    # --- Normalize + save per name ---
    output_files: list[str] = []
    per_name_summary: dict[str, Any] = {}
    for name, aligned_values in read.series.items():
        if normalize:
            if global_stats is not None:
                output_values, normalization = normalize_series_minmax_with_global_stats(
                    aligned_values, name, global_stats=global_stats
                )
            else:
                output_values, normalization = normalize_series_minmax(aligned_values, name)
        else:
            output_values = aligned_values.astype(np.float32, copy=False)
            normalization = _no_normalization_metadata([name])

        slug = _slugify_name(name)
        if torch_output:
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "PyTorch is required when --torch-output is enabled. Install `torch` first."
                ) from exc
            out_file = output_dir / f"injection_{slug}.pt"
            torch.save(
                {
                    "tensor": torch.tensor(output_values, dtype=torch.float32),
                    "name": name,
                    "time_ids": read.reference_months,
                    "time_ids_days": read.reference_days,
                    "time_unit": "months",
                    "normalization": normalization,
                    "source_path": str(input_file),
                    "sheet_name": sheet_name,
                },
                out_file,
            )
        else:
            out_file = output_dir / f"injection_{slug}.npz"
            np.savez_compressed(
                out_file,
                tensor=output_values,
                name=np.array([name]),
                time_ids=np.array(read.reference_months, dtype=np.int64),
                time_ids_days=np.array(read.reference_days, dtype=np.int64),
                time_unit=np.array(["months"]),
                normalization_method=np.array([normalization["method"]]),
                normalization_applied=np.array([int(bool(normalization.get("applied", False)))], dtype=np.int64),
            )

        output_files.append(str(out_file))
        dropped = read.dropped_months.get(name, [])
        per_name_summary[name] = {
            "length": int(output_values.shape[0]),
            "dropped_months_outside_reference": len(dropped),
            "dropped_months_preview": dropped[:5],
            "normalization": normalization,
        }

    report = {
        "mode": "injection-series",
        "input_path": str(input_file),
        "sheet_name": sheet_name,
        "selected_names": sorted(read.series.keys()),
        "selected_parameter": read.selected_parameter,
        "parameter_column_present": read.has_parameter_column,
        "available_parameters_preview": read.available_parameters_preview,
        "output_dir": str(output_dir),
        "output_format": "pt" if torch_output else "npz",
        "output_files_count": len(output_files),
        "output_files_preview": output_files[:3],
        "time_ids": _preview(read.reference_months, limit=REPORT_TIME_IDS_PREVIEW_LIMIT),
        "time_ids_days": _preview(read.reference_days, limit=REPORT_TIME_IDS_PREVIEW_LIMIT),
        "time_ids_count": len(read.reference_months),
        "time_ids_days_count": len(read.reference_days),
        "time_ids_preview_limit": REPORT_TIME_IDS_PREVIEW_LIMIT,
        "time_unit": "months",
        "duplicates_aggregated": read.duplicate_rows,
        "series": per_name_summary,
        "execution_seconds": round(perf_counter() - start, 6),
    }
    report_path = processed_path / "injection_name_tensors_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_layer_cubes_pipeline(
    sf_path: str | Path = "data/raw/SF_RED.txt",
    vd_path: str | Path = "data/raw/VD_RED.txt",
    *,
    NZ: int = 20,
    NJ: int = 100,
    NI: int = 100,
    processed_dir: str | Path = "data/processed",
    normalize: bool = True,
    full_tensor_output: bool = False,
    torch_output: bool = True,
    global_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build NZ cubes (2, T, NJ, NI) from SF + VD files, variable order [SF, VD]."""
    start = perf_counter()
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    layers_dir = processed_path / "layer_cubes"
    layers_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_outputs(layers_dir)

    cubes, time_ids_days = build_layer_cubes(
        sf_path=sf_path, vd_path=vd_path,
        NZ=NZ, NJ=NJ, NI=NI,
        dtype=np.float32, torch_output=False, return_times=True,
    )
    variable_order = ["SF", "VD"]
    cubes, normalization = _normalize_or_cast_cubes(cubes, variable_order, normalize, global_stats=global_stats)
    time_ids = _days_to_month_ids(time_ids_days)
    timeline_path = _write_timeline_file(
        processed_path, time_ids=time_ids, time_ids_days=time_ids_days, filename="timeline_sf_vd.json",
    )
    output_files, layer_cubes_saved, full_tensor_path, full_tensor_shape = _save_layer_or_full_outputs(
        cubes=cubes, time_ids=time_ids, time_ids_days=time_ids_days, variable_order=variable_order,
        normalization=normalization, layers_dir=layers_dir, processed_path=processed_path,
        full_tensor_output=full_tensor_output, torch_output=torch_output, full_tensor_stem="sf_vd_tensor",
    )

    cube_shape = tuple(cubes[0].shape) if cubes else None
    report = {
        "mode": "layer-cubes",
        "sf_path": str(sf_path),
        "vd_path": str(vd_path),
        "output_dir": str(layers_dir),
        "output_format": "pt" if torch_output else "npz",
        "output_files_count": len(output_files),
        "output_files_preview": output_files[:3],
        "layer_cubes_saved": layer_cubes_saved,
        "nz": NZ,
        "num_layers": len(cubes),
        "cube_shape": cube_shape,
        "timeline_path": str(timeline_path.relative_to(processed_path)),
        "time_ids": _preview(time_ids, limit=REPORT_TIME_IDS_PREVIEW_LIMIT),
        "time_ids_days": _preview(time_ids_days, limit=REPORT_TIME_IDS_PREVIEW_LIMIT),
        "time_ids_count": len(time_ids),
        "time_ids_days_count": len(time_ids_days),
        "time_ids_preview_limit": REPORT_TIME_IDS_PREVIEW_LIMIT,
        "time_unit": "months",
        "variable_order": variable_order,
        "normalization": normalization,
        "full_tensor_output": bool(full_tensor_output),
        "full_tensor_path": str(full_tensor_path) if full_tensor_path else None,
        "full_tensor_shape": full_tensor_shape,
        "execution_seconds": round(perf_counter() - start, 6),
    }
    report_path = processed_path / "layer_cubes_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_single_variable_layer_cubes_pipeline(
    variable_name: str,
    input_path: str | Path,
    *,
    NZ: int,
    NJ: int,
    NI: int,
    processed_dir: str | Path = "data/processed",
    normalize: bool = True,
    full_tensor_output: bool = False,
    torch_output: bool = True,
    strict: bool = True,
    global_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build NZ cubes (1, T, NJ, NI) for one variable from a single input file."""
    start = perf_counter()
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    layers_dir = processed_path / f"{variable_name.lower()}_layer_cubes"
    layers_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_outputs(layers_dir)

    cubes, time_ids_days = build_single_variable_layer_cubes(
        path=input_path,
        NZ=NZ, NJ=NJ, NI=NI,
        dtype=np.float32, torch_output=False, return_times=True, strict=strict,
    )

    variable_order = [variable_name.upper()]
    cubes, normalization = _normalize_or_cast_cubes(cubes, variable_order, normalize, global_stats=global_stats)
    time_ids = _days_to_month_ids(time_ids_days)
    timeline_path = _write_timeline_file(
        processed_path, time_ids=time_ids, time_ids_days=time_ids_days,
        filename=f"timeline_{variable_name.lower()}.json",
    )
    output_files, layer_cubes_saved, full_tensor_path, full_tensor_shape = _save_layer_or_full_outputs(
        cubes=cubes, time_ids=time_ids, time_ids_days=time_ids_days, variable_order=variable_order,
        normalization=normalization, layers_dir=layers_dir, processed_path=processed_path,
        full_tensor_output=full_tensor_output, torch_output=torch_output,
        full_tensor_stem=f"{variable_name.lower()}_tensor",
    )

    cube_shape = tuple(cubes[0].shape) if cubes else None
    report = {
        "mode": "single-variable-layer-cubes",
        "variable": variable_name.lower(),
        "input_path": str(input_path),
        "output_dir": str(layers_dir),
        "output_format": "pt" if torch_output else "npz",
        "output_files_count": len(output_files),
        "output_files_preview": output_files[:3],
        "layer_cubes_saved": layer_cubes_saved,
        "nz": NZ,
        "num_layers": len(cubes),
        "cube_shape": cube_shape,
        "timeline_path": str(timeline_path.relative_to(processed_path)),
        "time_ids": _preview(time_ids, limit=REPORT_TIME_IDS_PREVIEW_LIMIT),
        "time_ids_days": _preview(time_ids_days, limit=REPORT_TIME_IDS_PREVIEW_LIMIT),
        "time_ids_count": len(time_ids),
        "time_ids_days_count": len(time_ids_days),
        "time_ids_preview_limit": REPORT_TIME_IDS_PREVIEW_LIMIT,
        "time_unit": "months",
        "variable_order": variable_order,
        "normalization": normalization,
        "full_tensor_output": bool(full_tensor_output),
        "full_tensor_path": str(full_tensor_path) if full_tensor_path else None,
        "full_tensor_shape": full_tensor_shape,
        "execution_seconds": round(perf_counter() - start, 6),
    }
    report_path = processed_path / f"{variable_name.lower()}_layer_cubes_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _run_requested_pipelines(
    *,
    sf_path: str | Path | None,
    vd_path: str | Path | None,
    permeability_path: str | Path | None,
    porosity_path: str | Path | None,
    cohesion_path: str | Path | None,
    afi_path: str | Path | None,
    pressure_path: str | Path | None,
    gas_saturation_path: str | Path | None,
    injection_path: str | Path | None,
    injection_sheet: str,
    processed_dir: str | Path,
    nz: int,
    nj: int,
    ni: int,
    normalize: bool,
    full_tensor_output: bool,
    torch_output: bool,
    global_stats: dict[str, dict[str, float]] | None = None,
    skip_existing_pipelines: bool = False,
) -> dict[str, Any]:
    """Orchestrate all pipelines for one simulation directory."""
    has_sf = sf_path is not None
    has_vd = vd_path is not None
    if has_sf != has_vd:
        raise ValueError("Provide both --sf-path and --vd-path, or provide neither.")

    reports: dict[str, Any] = {}
    processed_path = Path(processed_dir)

    if has_sf and has_vd:
        sf_vd_report_path = processed_path / "layer_cubes_report.json"
        if skip_existing_pipelines and sf_vd_report_path.exists():
            reports["sf_vd"] = {"skipped": True, "reason": "existing_report"}
        else:
            sf_vd_report = run_layer_cubes_pipeline(
                sf_path=sf_path, vd_path=vd_path,
                NZ=nz, NJ=nj, NI=ni,
                processed_dir=processed_dir,
                normalize=normalize, full_tensor_output=full_tensor_output, torch_output=torch_output,
                global_stats=global_stats,
            )
            _print_cube_dimensions(sf_vd_report, "layer-cubes")
            _print_execution_time("sf_vd", sf_vd_report["execution_seconds"])
            reports["sf_vd"] = sf_vd_report

    single_variable_inputs: list[tuple[str, str | Path | None, bool]] = [
        ("permeability", permeability_path, True),
        ("porosity", porosity_path, True),
        ("cohesion", cohesion_path, True),
        ("afi", afi_path, True),
        ("pressure", pressure_path, True),
        ("gas_saturation", gas_saturation_path, False),
    ]
    for variable_name, variable_path, strict in single_variable_inputs:
        if not variable_path:
            continue
        variable_report_path = processed_path / f"{variable_name.lower()}_layer_cubes_report.json"
        if skip_existing_pipelines and variable_report_path.exists():
            reports[variable_name] = {"skipped": True, "reason": "existing_report"}
            continue
        variable_report = run_single_variable_layer_cubes_pipeline(
            variable_name, variable_path,
            NZ=nz, NJ=nj, NI=ni,
            processed_dir=processed_dir,
            normalize=normalize, full_tensor_output=full_tensor_output,
            torch_output=torch_output, strict=strict,
            global_stats=global_stats,
        )
        _print_cube_dimensions(variable_report, f"{variable_name}-layer-cubes")
        _print_execution_time(variable_name, variable_report["execution_seconds"])
        reports[variable_name] = variable_report

    if injection_path:
        injection_report_path = processed_path / "injection_name_tensors_report.json"
        if skip_existing_pipelines and injection_report_path.exists():
            reports["injection"] = {"skipped": True, "reason": "existing_report"}
        else:
            reference_time_days = _load_reference_time_days_from_report(processed_dir)
            injection_report = run_injection_excel_pipeline(
                injection_path,
                reference_time_days=reference_time_days,
                sheet_name=injection_sheet,
                include_names=("TENE-1", "TENE-2"),
                processed_dir=processed_dir,
                normalize=normalize, torch_output=torch_output,
                global_stats=global_stats,
            )
            timesteps = int(injection_report.get("time_ids_count") or len(injection_report["time_ids"]))
            print(
                "[DIM] injection-series: "
                f"{len(injection_report['series'])} nombres, "
                f"{timesteps} timesteps por serie (T)"
            )
            _print_execution_time("injection", injection_report["execution_seconds"])
            reports["injection"] = injection_report

    if not reports:
        raise ValueError(
            "No inputs provided. Use --sf-path/--vd-path and/or one of: "
            "--permeability-path, --porosity-path, --cohesion-path, --afi-path, "
            "--pressure-path, --gas-saturation-path, --injection-path."
        )

    return reports
