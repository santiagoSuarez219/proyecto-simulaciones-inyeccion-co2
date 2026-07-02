"""
Min-max normalization utilities for cmg2tensor cubes and time series.

These functions are extracted from app.py so they can be reused by
pipeline_parallel.py workers without importing the full CLI module.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def clip_percentiles(
    values: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Recorta valores fuera de [percentil p_low, percentil p_high] (B3:
    robust scaling / mitigar sensibilidad a outliers del min-max plano).

    Uso opcional ANTES de normalize_cubes_minmax*/normalize_series_minmax* —
    NO se integra automaticamente en el pipeline: cambiar el comportamiento
    default de normalizacion requeriria reprocesar datos reales (igual que
    C1), fuera de alcance sin confirmacion explicita y sin datos en esta
    sesion. Queda disponible para quien quiera evaluarla explicitamente."""
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    lo = np.percentile(finite, p_low)
    hi = np.percentile(finite, p_high)
    return np.clip(values, lo, hi)


def normalize_cubes_minmax(
    cubes: list[np.ndarray],
    variable_order: list[str],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Normalize a list of layer cubes using local (per-batch) min-max scaling.

    Each cube has shape (V, T, NJ, NI). Global min/max are computed across
    ALL cubes for each variable index, then each cube is normalized in-place
    on a copy.

    Returns:
        (normalized_cubes, normalization_metadata)
    """
    if not cubes:
        return cubes, {"applied": False, "method": "minmax", "per_variable": {}}

    num_vars = cubes[0].shape[0]
    mins = np.full(num_vars, np.inf, dtype=np.float64)
    maxs = np.full(num_vars, -np.inf, dtype=np.float64)

    for cube in cubes:
        for var_idx in range(num_vars):
            var_data = cube[var_idx]
            mins[var_idx] = min(mins[var_idx], float(np.nanmin(var_data)))
            maxs[var_idx] = max(maxs[var_idx], float(np.nanmax(var_data)))

    per_variable: dict[str, Any] = {}
    spans: list[float] = []
    for var_idx in range(num_vars):
        var_name = variable_order[var_idx] if var_idx < len(variable_order) else f"var_{var_idx}"
        var_min = float(mins[var_idx])
        var_max = float(maxs[var_idx])
        var_span = float(var_max - var_min)
        spans.append(var_span)
        per_variable[var_name] = {
            "min": var_min,
            "max": var_max,
            "span": var_span,
        }

    normalized_cubes: list[np.ndarray] = []
    for cube in cubes:
        normalized = cube.astype(np.float32, copy=True)
        for var_idx, span in enumerate(spans):
            var_min = mins[var_idx]
            if span <= 0.0:
                normalized[var_idx] = 0.0
            else:
                normalized[var_idx] = (normalized[var_idx] - var_min) / span
        normalized_cubes.append(normalized)

    return normalized_cubes, {"applied": True, "method": "minmax", "per_variable": per_variable}


def normalize_cubes_minmax_with_global_stats(
    cubes: list[np.ndarray],
    variable_order: list[str],
    *,
    global_stats: dict[str, dict[str, float]],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Normalize a list of layer cubes using pre-computed global (train-split) min-max stats.

    Parameters
    ----------
    cubes:
        Each cube has shape (V, T, NJ, NI).
    variable_order:
        Variable names corresponding to axis 0 of each cube.
    global_stats:
        Dict mapping variable name → {min, max, span}.

    Returns:
        (normalized_cubes, normalization_metadata)
    """
    if not cubes:
        return cubes, {"applied": False, "method": "minmax", "per_variable": {}}

    per_variable: dict[str, Any] = {}
    mins: list[float] = []
    spans: list[float] = []
    for var_name in variable_order:
        stats = global_stats.get(var_name)
        if not stats:
            raise ValueError(f"Global normalization stats missing for variable '{var_name}'.")
        var_min = float(stats["min"])
        var_max = float(stats["max"])
        span = float(var_max - var_min)
        mins.append(var_min)
        spans.append(span)
        per_variable[var_name] = {"min": var_min, "max": var_max, "span": span}

    normalized_cubes: list[np.ndarray] = []
    for cube in cubes:
        normalized = cube.astype(np.float32, copy=True)
        for var_idx, span in enumerate(spans):
            var_min = mins[var_idx]
            if span <= 0.0:
                normalized[var_idx] = 0.0
            else:
                normalized[var_idx] = (normalized[var_idx] - var_min) / span
        normalized_cubes.append(normalized)

    return normalized_cubes, {
        "applied": True,
        "method": "minmax",
        "scope": "split",
        "per_variable": per_variable,
    }


def normalize_series_minmax(
    values: np.ndarray,
    series_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Normalize a 1D time series with local min-max scaling.

    Returns:
        (normalized_values, normalization_metadata)
    """
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    span = float(max_value - min_value)
    if span <= 0.0:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = ((values - min_value) / span).astype(np.float32, copy=False)
    return normalized, {
        "applied": True,
        "method": "minmax",
        "per_variable": {
            series_name: {
                "min": min_value,
                "max": max_value,
                "span": span,
            }
        },
    }


def normalize_series_minmax_with_global_stats(
    values: np.ndarray,
    series_name: str,
    *,
    global_stats: dict[str, dict[str, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Normalize a 1D time series using pre-computed global (train-split) min-max stats.

    Returns:
        (normalized_values, normalization_metadata)
    """
    stats = global_stats.get(series_name)
    if not stats:
        raise ValueError(f"Global normalization stats missing for series '{series_name}'.")
    min_value = float(stats["min"])
    max_value = float(stats["max"])
    span = float(max_value - min_value)
    if span <= 0.0:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = ((values - min_value) / span).astype(np.float32, copy=False)
    return normalized, {
        "applied": True,
        "method": "minmax",
        "scope": "split",
        "per_variable": {
            series_name: {
                "min": min_value,
                "max": max_value,
                "span": span,
            }
        },
    }
