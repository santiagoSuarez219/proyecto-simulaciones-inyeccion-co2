from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_STANDARDIZE_LOG_FILE = DEFAULT_REPORTS_DIR / "standardization_log.txt"
DEFAULT_RAW_STRUCTURE_REPORT_FILE = DEFAULT_REPORTS_DIR / "raw_structure_report.txt"


def ensure_src_on_path() -> None:
    """Add src/ to sys.path when running helper scripts directly."""
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


# Import the canonical discovery implementation.
# Lazy import so this module can be used without installing the package.
def _get_discover_raw_roles():
    try:
        from cmg2tensor.discovery import discover_raw_roles
        return discover_raw_roles
    except ImportError:
        ensure_src_on_path()
        from cmg2tensor.discovery import discover_raw_roles
        return discover_raw_roles


def discover_raw_roles(sim_dir: Path) -> dict[str, Path | None]:
    """
    Identify raw files by "role" using filename heuristics.

    Delegates to cmg2tensor.discovery.discover_raw_roles — single source of truth.
    """
    return _get_discover_raw_roles()(sim_dir)


@dataclass(frozen=True)
class RenameAction:
    src: Path
    dst: Path
    reason: str


def _plan_standardization(sim_dir: Path) -> tuple[list[RenameAction], list[Path]]:
    roles = discover_raw_roles(sim_dir)
    desired_names: dict[str, str] = {
        "sf": "SF.txt",
        "vd": "VD.txt",
        "cohesion": "cohesion.txt",
        "friction_angle": "friction_angle.txt",
        "inyeccion": "inyeccion.xlsx",
        "pressure": "pressure.txt",
        "gas_saturation": "gas_saturation.txt",
    }

    actions: list[RenameAction] = []
    used_sources = {p for p in roles.values() if p is not None}
    unknown_files = [p for p in sim_dir.iterdir() if p.is_file() and p not in used_sources]

    for role, src_path in roles.items():
        if src_path is None:
            continue
        desired = sim_dir / desired_names[role]
        if src_path.name == desired.name:
            continue
        actions.append(RenameAction(src=src_path, dst=desired, reason=role))

    return actions, sorted(unknown_files, key=lambda p: p.name.lower())


def _format_action(action: RenameAction) -> str:
    return f"{action.src.name} -> {action.dst.name} ({action.reason})"


def _unique_backup_path(path: Path) -> Path:
    candidate = path.with_suffix(path.suffix + ".bak")
    if not candidate.exists():
        return candidate
    for idx in range(1, 10_000):
        candidate = path.with_suffix(path.suffix + f".bak{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find free backup name for: {path}")


def _iter_sim_dirs(raw_root: Path, *, include_v2: bool) -> list[Path]:
    sim_dirs = [p for p in raw_root.iterdir() if p.is_dir()]
    if not include_v2:
        sim_dirs = [p for p in sim_dirs if p.name.lower() != "v_2"]

    prefix_re = re.compile(r"^(?P<id>\d{1,3})_")

    def _sort_key(p: Path):
        if p.name.isdigit():
            return (0, int(p.name), p.name.lower())
        match = prefix_re.match(p.name)
        if match:
            return (0, int(match.group("id")), p.name.lower())
        return (1, 10**9, p.name.lower())

    return sorted(sim_dirs, key=_sort_key)


def standardize_raw_names(
    *,
    raw_root: Path,
    apply_changes: bool,
    log_file: Path,
    include_v2: bool,
    overwrite: bool,
) -> int:
    if not raw_root.exists():
        raise SystemExit(f"raw_root does not exist: {raw_root}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    sim_dirs = _iter_sim_dirs(raw_root, include_v2=include_v2)

    changed = 0
    warnings = 0

    with log_file.open("w", encoding="utf-8-sig") as log:
        log.write(f"raw_root: {raw_root}\n")
        log.write(f"mode: {'APPLY' if apply_changes else 'DRY_RUN'}\n\n")

        for sim_dir in sim_dirs:
            actions, unknown_files = _plan_standardization(sim_dir)
            roles = discover_raw_roles(sim_dir)

            missing_required: list[str] = []
            if roles["sf"] is None:
                missing_required.append("SF")
            if roles["vd"] is None:
                missing_required.append("VD")

            log.write(f"=== {sim_dir.name} ===\n")
            if missing_required:
                warnings += 1
                log.write(f"WARNING: missing required: {', '.join(missing_required)}\n")

            if not actions:
                log.write("OK: no renames needed\n")
            else:
                for action in actions:
                    dst_exists = action.dst.exists()
                    if dst_exists and not overwrite:
                        warnings += 1
                        log.write(f"SKIP (exists): {_format_action(action)}\n")
                        continue

                    if apply_changes:
                        if dst_exists and overwrite:
                            backup = _unique_backup_path(action.dst)
                            warnings += 1
                            log.write(f"OVERWRITE: moving existing {action.dst.name} -> {backup.name}\n")
                            action.dst.rename(backup)

                        action.src.rename(action.dst)
                        changed += 1
                        log.write(f"RENAMED: {_format_action(action)}\n")
                    else:
                        log.write(f"PLAN: {_format_action(action)}\n")

            if unknown_files:
                warnings += 1
                log.write("WARNING: unrecognized files:\n")
                for path in unknown_files:
                    log.write(f"  - {path.name}\n")
            log.write("\n")

        log.write(f"summary_changed: {changed}\n")
        log.write(f"summary_warnings: {warnings}\n")

    print(f"[OK] Log: {log_file}")
    if not apply_changes:
        print("[INFO] Dry-run mode: nothing was renamed. Use --apply to rename.")
    if warnings:
        print(f"[WARN] {warnings} warnings. Check the log for details.")
    return 0 if warnings == 0 else 2


def _format_role(role: str, path: Path | None) -> str:
    return f"{role}={path.name}" if path else f"{role}=MISSING"


def _present_roles(roles: dict[str, Path | None]) -> str:
    ordered = ["sf", "vd", "cohesion", "friction_angle", "pressure", "gas_saturation", "inyeccion"]
    return ", ".join(_format_role(role, roles.get(role)) for role in ordered)


def write_raw_structure_report(*, raw_root: Path, out_file: Path, include_v2: bool) -> int:
    if not raw_root.exists():
        raise SystemExit(f"raw_root does not exist: {raw_root}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    sim_dirs = _iter_sim_dirs(raw_root, include_v2=include_v2)
    if not sim_dirs:
        raise SystemExit(f"No simulation directories found in: {raw_root}")

    role_sets: dict[str, set[str]] = {}
    missing_required: list[str] = []

    lines: list[str] = []
    lines.append(f"raw_root: {raw_root}")
    lines.append("")
    lines.append("=== Per-simulation summary ===")

    for sim_dir in sim_dirs:
        roles = discover_raw_roles(sim_dir)
        present_roles = {k for k, v in roles.items() if v is not None}
        role_sets[sim_dir.name] = present_roles

        missing = []
        if roles.get("sf") is None:
            missing.append("SF")
        if roles.get("vd") is None:
            missing.append("VD")
        if missing:
            missing_required.append(sim_dir.name)

        file_count = len([p for p in sim_dir.iterdir() if p.is_file()])
        lines.append(
            f"- {sim_dir.name}: files={file_count} | roles={_present_roles(roles)}"
            + (f" | missing_required={','.join(missing)}" if missing else "")
        )

    lines.append("")
    lines.append("=== Consistency ===")
    unique_role_sets = {tuple(sorted(v)) for v in role_sets.values()}
    if len(unique_role_sets) == 1:
        lines.append(f"OK: all folders share the same role set: {sorted(next(iter(unique_role_sets)))}")
    else:
        lines.append("WARN: not all folders share the same role set.")

        def _sim_sort_key(name: str) -> int:
            return int(name) if name.isdigit() else 10**9

        for sim_name in sorted(role_sets.keys(), key=_sim_sort_key):
            lines.append(f"- {sim_name}: {sorted(role_sets[sim_name])}")

    lines.append("")
    lines.append("=== Required files (SF/VD) ===")
    if not missing_required:
        lines.append("OK: all folders have SF and VD.")
    else:
        lines.append("WARN: some folders are missing SF and/or VD:")
        for name in missing_required:
            lines.append(f"- {name}")

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"[OK] Report: {out_file}")
    return 0 if not missing_required and len(unique_role_sets) == 1 else 2


def build_standardize_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardize file names inside data/raw/<simulation> folders.",
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_STANDARDIZE_LOG_FILE)
    parser.add_argument("--apply", action="store_true", help="Apply renames (default: dry-run).")
    parser.add_argument("--include-v2", action="store_true", help="Also process data/raw/v_2.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If destination exists, move it to .bak and overwrite.",
    )
    return parser


def build_check_formats_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check structure of data/raw/* simulation folders.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_RAW_STRUCTURE_REPORT_FILE)
    parser.add_argument("--include-v2", action="store_true")
    return parser
