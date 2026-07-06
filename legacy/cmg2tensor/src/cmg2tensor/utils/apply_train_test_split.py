from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


from cmg2tensor.config import DEFAULT_SPLIT_CSV as _SPLIT_CSV_REL

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_CSV = PROJECT_ROOT / _SPLIT_CSV_REL
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "processed" / "Simulaciones"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "apply_train_test_split_report.json"


def _read_split_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"split csv not found: {path}")
    split_by_name: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"simulation_name", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {path.name}: {sorted(missing)}")
        for row in reader:
            name = str(row["simulation_name"]).strip()
            split = str(row["split"]).strip().lower()
            if not name or split not in {"train", "test"}:
                continue
            split_by_name[name] = split
    if not split_by_name:
        raise SystemExit(f"No valid train/test rows found in: {path}")
    return split_by_name


def apply_split(
    *,
    split_csv: Path,
    source_root: Path,
    apply_changes: bool,
    report_json: Path,
    move_unassigned: str,
) -> int:
    split_by_name = _read_split_csv(split_csv)
    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit(f"source_root not found or not a directory: {source_root}")

    train_dir = source_root / "train"
    test_dir = source_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "apply": bool(apply_changes),
        "split_csv": str(split_csv),
        "source_root": str(source_root),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "move_unassigned": move_unassigned,
        "moved": [],
        "skipped": [],
        "missing": [],
        "unassigned": [],
    }

    # Only move simulation dirs that already exist at source_root/<sim>.
    existing_sim_dirs = {p.name: p for p in source_root.iterdir() if p.is_dir()}
    # ignore train/test dirs themselves
    existing_sim_dirs.pop("train", None)
    existing_sim_dirs.pop("test", None)

    moved_names: set[str] = set()
    for sim_name, split in sorted(split_by_name.items(), key=lambda kv: kv[0].lower()):
        src = existing_sim_dirs.get(sim_name)
        if src is None:
            report["missing"].append({"simulation": sim_name, "split": split})
            continue

        dst_parent = train_dir if split == "train" else test_dir
        dst = dst_parent / sim_name
        if dst.exists():
            report["skipped"].append(
                {"simulation": sim_name, "split": split, "reason": "dst_exists", "dst": str(dst)}
            )
            continue

        report["moved"].append(
            {"simulation": sim_name, "split": split, "src": str(src), "dst": str(dst), "applied": bool(apply_changes)}
        )
        if apply_changes:
            src.rename(dst)
        moved_names.add(sim_name)

    # Optionally move any remaining dirs that were not present in split CSV.
    if move_unassigned != "skip":
        for sim_name, src in sorted(existing_sim_dirs.items(), key=lambda kv: kv[0].lower()):
            if sim_name in moved_names:
                continue
            dst_parent = train_dir if move_unassigned == "train" else test_dir
            dst = dst_parent / sim_name
            if dst.exists():
                report["skipped"].append(
                    {
                        "simulation": sim_name,
                        "split": "unassigned",
                        "reason": "dst_exists",
                        "dst": str(dst),
                    }
                )
                continue
            report["unassigned"].append(
                {
                    "simulation": sim_name,
                    "moved_to": move_unassigned,
                    "src": str(src),
                    "dst": str(dst),
                    "applied": bool(apply_changes),
                }
            )
            if apply_changes:
                src.rename(dst)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    moved = len(report["moved"])
    missing = len(report["missing"])
    skipped = len(report["skipped"])
    print(f"[OK] Report: {report_json}")
    print(f"[INFO] moved={moved} skipped={skipped} missing={missing} apply={bool(apply_changes)}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Move simulation folders into train/ and test/ according to split CSV.")
    p.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--apply", action="store_true", help="Apply moves (default: dry-run).")
    p.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    p.add_argument(
        "--move-unassigned",
        choices=("skip", "train", "test"),
        default="skip",
        help="What to do with simulation folders in source-root that are not present in split CSV.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return apply_split(
        split_csv=args.split_csv,
        source_root=args.source_root,
        apply_changes=bool(args.apply),
        report_json=args.report_json,
        move_unassigned=str(args.move_unassigned),
    )


if __name__ == "__main__":
    raise SystemExit(main())
