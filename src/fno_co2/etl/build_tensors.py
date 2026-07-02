from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .parse_txt import ParseResult


def build_tensor_payload(parse_result: ParseResult, dtype: Any = None) -> dict[str, Any]:
    """Build a serializable payload with tensor and aligned time metadata."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to build `.pt` outputs. Install `torch` in your active environment."
        ) from exc

    if dtype is None:
        dtype = torch.float32
    tensor = torch.tensor(parse_result.cube_4d, dtype=dtype)
    return {
        "tensor": tensor,
        "time_days": parse_result.time_days,
        "time_dates": parse_result.time_dates,
    }


def save_tensor_pt(
    parse_result: ParseResult,
    output_path: str | Path,
    *,
    dtype: Any = None,
) -> Path:
    """Save tensor payload to disk as a `.pt` file."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to save `.pt` outputs. Install `torch` in your active environment."
        ) from exc

    payload = build_tensor_payload(parse_result, dtype=dtype)
    destination = Path(output_path)
    torch.save(payload, destination)
    return destination


def save_time_metadata_csv(
    time_days: list[int],
    time_dates: list[str],
    csv_path: str | Path,
) -> Path:
    """Save time metadata using the same temporal index as the tensor."""
    destination = Path(csv_path)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_index", "time_days", "time_dates"])
        for idx, (day, date) in enumerate(zip(time_days, time_dates)):
            writer.writerow([idx, day, date])
    return destination


def summarize_tensor(cube_4d: np.ndarray) -> dict[str, Any]:
    """Compute lightweight summary stats for quick validation."""
    return {
        "shape": tuple(cube_4d.shape),
        "nan_count": int(np.isnan(cube_4d).sum()),
        "global_min": float(np.nanmin(cube_4d)),
        "global_max": float(np.nanmax(cube_4d)),
    }
