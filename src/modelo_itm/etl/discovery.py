"""
Simulation input discovery for cmg2tensor.

Provides a single, authoritative implementation of filename-heuristic matching
used by both the processing pipeline and the raw-standardization utilities.

Public API
----------
discover_simulation_inputs(sim_dir)
    Returns the canonical {role: Path|None} dict used by the pipeline.
    Raises ValueError if SF or VD are missing.

discover_raw_roles(sim_dir)
    Returns the same dict without raising on missing files.
    Used by raw_standardization to plan renames and write structure reports.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Low-level filename helpers
# ---------------------------------------------------------------------------

def _tokenize_stem(path: Path) -> set[str]:
    """Split a filename stem into lowercase ASCII tokens (strips diacritics)."""
    stem = path.stem.strip().lower()
    normalized = unicodedata.normalize("NFKD", stem)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return {token for token in re.split(r"[^a-z0-9]+", normalized) if token}


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _pick_first_matching_file(
    files: list[Path],
    *,
    required_tokens: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> Path | None:
    """Return the first file whose stem contains ALL required_tokens and has an allowed suffix."""
    allowed = {sfx.lower() for sfx in suffixes}
    for p in sorted(files, key=lambda f: f.name.lower()):
        if not _is_nonempty_file(p):
            continue
        if p.suffix.lower() not in allowed:
            continue
        if all(tok in _tokenize_stem(p) for tok in required_tokens):
            return p
    return None


def _pick_first_matching_file_any(
    files: list[Path],
    *,
    required_token_groups: list[tuple[str, ...]],
    suffixes: tuple[str, ...],
    check_nonempty: bool = True,
) -> Path | None:
    """Return the first file that matches ANY of the required_token_groups."""
    allowed = {sfx.lower() for sfx in suffixes}
    for p in sorted(files, key=lambda f: f.name.lower()):
        if check_nonempty and not _is_nonempty_file(p):
            continue
        if p.suffix.lower() not in allowed:
            continue
        tokens = _tokenize_stem(p)
        if any(all(tok in tokens for tok in group) for group in required_token_groups):
            return p
    return None


# ---------------------------------------------------------------------------
# Public discovery functions
# ---------------------------------------------------------------------------

def discover_simulation_inputs(sim_dir: Path) -> dict[str, Path | None]:
    """
    Discover all input files for one simulation directory.

    Returns a dict with keys:
        sf_path, vd_path, permeability_path, porosity_path, cohesion_path,
        afi_path, pressure_path, gas_saturation_path, injection_path

    Raises ValueError if SF or VD are missing (required by the pipeline).
    """
    files = [p for p in sim_dir.iterdir() if p.is_file()]
    if not files:
        raise ValueError(f"Simulation folder has no files: {sim_dir}")

    sf_path = _pick_first_matching_file_any(
        files,
        required_token_groups=[("sf",), ("safety", "factor")],
        suffixes=(".txt",),
    )
    vd_path = _pick_first_matching_file_any(
        files,
        required_token_groups=[("vd",), ("vertical", "displacement")],
        suffixes=(".txt",),
    )
    permeability_path = _pick_first_matching_file(
        files, required_tokens=("permeability",), suffixes=(".txt",),
    )
    porosity_path = _pick_first_matching_file(
        files, required_tokens=("porosity",), suffixes=(".txt",),
    )
    cohesion_path = _pick_first_matching_file(
        files, required_tokens=("cohesion",), suffixes=(".txt",),
    )
    afi_path = _pick_first_matching_file_any(
        files,
        required_token_groups=[
            ("afi",),
            ("friction", "angle"),
            ("friccion", "angle"),
            ("friccion", "angulo"),
        ],
        suffixes=(".txt",),
    )
    pressure_path = _pick_first_matching_file(
        files, required_tokens=("pressure",), suffixes=(".txt",),
    )
    gas_saturation_path = _pick_first_matching_file_any(
        files,
        required_token_groups=[("gas", "saturation"), ("sg",)],
        suffixes=(".txt",),
    )
    injection_path = _pick_first_matching_file(
        files, required_tokens=("inyeccion",), suffixes=(".xlsx", ".xls"),
    )
    if injection_path is None:
        injection_path = _pick_first_matching_file(
            files, required_tokens=("injection",), suffixes=(".xlsx", ".xls"),
        )

    missing = []
    if sf_path is None:
        missing.append("SF (*.txt)")
    if vd_path is None:
        missing.append("VD (*.txt)")
    if missing:
        raise ValueError(
            f"Simulation '{sim_dir.name}' is missing required files: {', '.join(missing)}"
        )

    return {
        "sf_path": sf_path,
        "vd_path": vd_path,
        "permeability_path": permeability_path,
        "porosity_path": porosity_path,
        "cohesion_path": cohesion_path,
        "afi_path": afi_path,
        "pressure_path": pressure_path,
        "gas_saturation_path": gas_saturation_path,
        "injection_path": injection_path,
    }


def discover_raw_roles(sim_dir: Path) -> dict[str, Path | None]:
    """
    Identify raw files by role for the standardization utility.

    Like discover_simulation_inputs but:
    - Does NOT raise on missing SF/VD (caller decides policy).
    - Uses check_nonempty=False so empty/stub files are still matched.
    - Returns role keys aligned with desired_names in raw_standardization.py.
    """
    files = [p for p in sim_dir.iterdir() if p.is_file()]
    return {
        "sf": _pick_first_matching_file_any(
            files,
            required_token_groups=[("sf",), ("safety", "factor")],
            suffixes=(".txt",),
            check_nonempty=False,
        ),
        "vd": _pick_first_matching_file_any(
            files,
            required_token_groups=[("vd",), ("vertical", "displacement")],
            suffixes=(".txt",),
            check_nonempty=False,
        ),
        "cohesion": _pick_first_matching_file_any(
            files,
            required_token_groups=[("cohesion",)],
            suffixes=(".txt",),
            check_nonempty=False,
        ),
        # "friction_angle" / "afi" are the same physical variable.
        # raw_standardization uses key "friction_angle" for the desired rename target.
        "friction_angle": _pick_first_matching_file_any(
            files,
            required_token_groups=[
                ("afi",),
                ("friction", "angle"),
                ("friccion", "angle"),
                ("friccion", "angulo"),
            ],
            suffixes=(".txt",),
            check_nonempty=False,
        ),
        "inyeccion": _pick_first_matching_file_any(
            files,
            required_token_groups=[("inyeccion",), ("injection",), ("tasa", "inyeccion")],
            suffixes=(".xlsx", ".xls"),
            check_nonempty=False,
        ),
        "pressure": _pick_first_matching_file_any(
            files,
            required_token_groups=[("pressure",)],
            suffixes=(".txt",),
            check_nonempty=False,
        ),
        "gas_saturation": _pick_first_matching_file_any(
            files,
            required_token_groups=[("gas", "saturation"), ("sg",)],
            suffixes=(".txt",),
            check_nonempty=False,
        ),
    }
