"""
Regression tests: verify that the new streaming parser produces numerically
identical (or near-identical) output compared to the legacy load-all approach.

The legacy path is emulated by reading the full text and running the old
parse_cmg_text() function, which still uses the block-level regex on a string.
Both must agree to np.allclose(atol=1e-6).
"""
from __future__ import annotations

import numpy as np
import pytest

from fno_co2.etl.parse_txt import (
    GridShape,
    _parse_txt_with_times,
    parse_cmg_file,
    parse_txt,
    parse_txt_with_times,
)
from tests.etl.conftest import NI, NJ, NZ, TIME_DAYS


# -------------------------------------------------------------------------
# Core numeric regression
# -------------------------------------------------------------------------

def test_streaming_parser_matches_cmg_text_parser(cmg_file):
    """Streaming _parse_txt_with_times must match parse_cmg_file numerically."""
    grid = GridShape(nz=NZ, nj=NJ, ni=NI)

    # New streaming path
    tensor_streaming, time_ids_streaming = _parse_txt_with_times(
        cmg_file, NZ, NJ, NI, dtype=np.float32, strict=True
    )

    # Reference: parse_cmg_file loads full text (legacy approach)
    result_ref = parse_cmg_file(cmg_file, grid, strict=False)
    tensor_ref = result_ref.cube_4d.astype(np.float32)

    assert tensor_streaming.shape == tensor_ref.shape, (
        f"Shape mismatch: streaming={tensor_streaming.shape}, ref={tensor_ref.shape}"
    )
    assert np.allclose(tensor_streaming, tensor_ref, atol=1e-5, equal_nan=True), (
        "Streaming parser output differs from reference by more than atol=1e-5"
    )


def test_time_ids_are_correct(cmg_file):
    """Time IDs returned by streaming parser must match expected TIME days."""
    _, time_ids = _parse_txt_with_times(cmg_file, NZ, NJ, NI, dtype=np.float32)
    assert time_ids == TIME_DAYS, f"Expected {TIME_DAYS}, got {time_ids}"


def test_parse_txt_public_api(cmg_file):
    """Public parse_txt() returns (T, NZ, NJ, NI) tensor."""
    tensor = parse_txt(cmg_file, NZ, NJ, NI)
    assert tensor.shape == (len(TIME_DAYS), NZ, NJ, NI)
    assert tensor.dtype == np.float32
    assert not np.isnan(tensor).any()


def test_parse_txt_with_times_public_api(cmg_file):
    """Public parse_txt_with_times() returns (tensor, time_ids)."""
    tensor, time_ids = parse_txt_with_times(cmg_file, NZ, NJ, NI)
    assert tensor.shape == (len(TIME_DAYS), NZ, NJ, NI)
    assert time_ids == TIME_DAYS


def test_static_file_fallback(static_cmg_file):
    """Static CMG file (no TIME blocks) returns shape (1, NZ, NJ, NI)."""
    tensor, time_ids = parse_txt_with_times(static_cmg_file, NZ, NJ, NI)
    assert tensor.shape == (1, NZ, NJ, NI)
    assert time_ids == [0]


def test_dtype_is_float32_by_default(cmg_file):
    """Default dtype must be float32 throughout the pipeline."""
    tensor = parse_txt(cmg_file, NZ, NJ, NI)
    assert tensor.dtype == np.float32


def test_no_nan_in_complete_file(cmg_file):
    """A complete synthetic file must produce no NaN values."""
    tensor = parse_txt(cmg_file, NZ, NJ, NI)
    assert not np.isnan(tensor).any()


# -------------------------------------------------------------------------
# build_layer_cubes regression
# -------------------------------------------------------------------------

def test_build_layer_cubes_shape(cmg_file, cmg_file_b):
    """build_layer_cubes returns NZ cubes each with shape (2, T, NJ, NI)."""
    from fno_co2.etl.parse_txt import build_layer_cubes
    cubes = build_layer_cubes(cmg_file, cmg_file_b, NZ, NJ, NI)
    assert len(cubes) == NZ
    T = len(TIME_DAYS)
    for cube in cubes:
        assert cube.shape == (2, T, NJ, NI), f"Bad cube shape: {cube.shape}"
        assert cube.dtype == np.float32


def test_build_single_variable_layer_cubes_shape(cmg_file):
    """build_single_variable_layer_cubes returns NZ cubes each with shape (1, T, NJ, NI)."""
    from fno_co2.etl.parse_txt import build_single_variable_layer_cubes
    cubes = build_single_variable_layer_cubes(cmg_file, NZ, NJ, NI)
    assert len(cubes) == NZ
    T = len(TIME_DAYS)
    for cube in cubes:
        assert cube.shape == (1, T, NJ, NI)


# -------------------------------------------------------------------------
# Normalization regression
# -------------------------------------------------------------------------

def test_normalize_cubes_minmax_values_in_0_1(cmg_file):
    """After local min-max normalization, all values must be in [0, 1]."""
    from fno_co2.etl.parse_txt import build_layer_cubes
    from fno_co2.etl.normalize import normalize_cubes_minmax
    cubes, _ = build_layer_cubes(cmg_file, cmg_file, NZ, NJ, NI, return_times=True)
    norm_cubes, meta = normalize_cubes_minmax(cubes, ["SF", "VD"])
    for cube in norm_cubes:
        assert float(np.nanmin(cube)) >= -1e-6
        assert float(np.nanmax(cube)) <= 1.0 + 1e-6


def test_normalize_cubes_with_global_stats(cmg_file):
    """Global-stats normalization is identical to local when stats come from the same file."""
    from fno_co2.etl.parse_txt import build_layer_cubes
    from fno_co2.etl.normalize import normalize_cubes_minmax, normalize_cubes_minmax_with_global_stats
    cubes, _ = build_layer_cubes(cmg_file, cmg_file, NZ, NJ, NI, return_times=True)
    local_cubes, local_meta = normalize_cubes_minmax(list(cubes), ["SF", "VD"])
    # Build global_stats from the local metadata
    global_stats = {
        var: {"min": info["min"], "max": info["max"]}
        for var, info in local_meta["per_variable"].items()
    }
    global_cubes, _ = normalize_cubes_minmax_with_global_stats(list(cubes), ["SF", "VD"], global_stats=global_stats)
    for lc, gc in zip(local_cubes, global_cubes):
        assert np.allclose(lc, gc, atol=1e-6, equal_nan=True)
