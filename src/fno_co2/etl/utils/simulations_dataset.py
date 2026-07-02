from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NOMENCLATURA_CSV = PROJECT_ROOT / "simulaciones_nomenclatura.csv"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "Simulaciones"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_SPLIT_CSV = DEFAULT_REPORTS_DIR / "train_test_split_80_20.csv"
DEFAULT_SPLIT_JSON = DEFAULT_REPORTS_DIR / "train_test_split_80_20.json"
DEFAULT_RENAME_REPORT_JSON = DEFAULT_REPORTS_DIR / "rename_simulations_report.json"


@dataclass(frozen=True)
class SimulationNomenclatureRow:
    global_id: int
    simulation_name: str
    set_name: str
    plan_type: str
    case_id: int | None = None


def _read_nomenclature_rows(path: Path) -> list[SimulationNomenclatureRow]:
    if not path.exists():
        raise SystemExit(f"nomenclatura csv not found: {path}")

    rows: list[SimulationNomenclatureRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"global_id", "simulation_name", "set", "plan_type"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {path.name}: {sorted(missing)}")

        for raw in reader:
            try:
                global_id = int(str(raw["global_id"]).strip())
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"Invalid global_id row: {raw}") from exc
            simulation_name = str(raw["simulation_name"]).strip()
            set_name = str(raw["set"]).strip()
            plan_type = str(raw["plan_type"]).strip()
            case_id = None
            if raw.get("case_id") not in (None, ""):
                try:
                    case_id = int(str(raw["case_id"]).strip())
                except Exception:
                    case_id = None
            rows.append(
                SimulationNomenclatureRow(
                    global_id=global_id,
                    simulation_name=simulation_name,
                    set_name=set_name,
                    plan_type=plan_type,
                    case_id=case_id,
                )
            )

    if not rows:
        raise SystemExit(f"No rows read from: {path}")
    return rows


def _build_index(rows: Iterable[SimulationNomenclatureRow]) -> dict[int, SimulationNomenclatureRow]:
    index: dict[int, SimulationNomenclatureRow] = {}
    duplicates: list[int] = []
    for row in rows:
        if row.global_id in index:
            duplicates.append(row.global_id)
            continue
        index[row.global_id] = row
    if duplicates:
        raise SystemExit(f"Duplicate global_id entries in nomenclatura csv: {sorted(set(duplicates))[:10]}")
    return index


_PREFIX_ID_RE = re.compile(r"^(?P<id>\d{1,3})_")


def _infer_global_id_from_dirname(name: str) -> int | None:
    if name.isdigit():
        return int(name)
    match = _PREFIX_ID_RE.match(name)
    if match:
        return int(match.group("id"))
    return None


def _rename_simulation_dirs(
    *,
    root: Path,
    index: dict[int, SimulationNomenclatureRow],
    apply_changes: bool,
    report: dict[str, Any],
    label: str,
) -> None:
    if not root.exists():
        report[label] = {"root": str(root), "exists": False, "actions": []}
        return
    if not root.is_dir():
        raise SystemExit(f"{label} root is not a directory: {root}")

    actions: list[dict[str, Any]] = []
    report[label] = {"root": str(root), "exists": True, "actions": actions}

    sim_dirs = [p for p in root.iterdir() if p.is_dir()]
    # deterministic order for logs
    sim_dirs = sorted(sim_dirs, key=lambda p: p.name.lower())

    for sim_dir in sim_dirs:
        old_name = sim_dir.name
        global_id = _infer_global_id_from_dirname(old_name)
        if global_id is None:
            actions.append({"src": old_name, "skipped": True, "reason": "unrecognized_dir_name"})
            continue

        row = index.get(global_id)
        if row is None:
            actions.append({"src": old_name, "skipped": True, "reason": "global_id_not_in_csv"})
            continue

        new_name = row.simulation_name
        if old_name == new_name:
            actions.append({"src": old_name, "dst": new_name, "skipped": True, "reason": "already_canonical"})
            continue

        dst = sim_dir.with_name(new_name)
        if dst.exists():
            actions.append(
                {
                    "src": old_name,
                    "dst": new_name,
                    "skipped": True,
                    "reason": "dst_exists",
                }
            )
            continue

        actions.append({"src": old_name, "dst": new_name, "applied": bool(apply_changes)})
        if apply_changes:
            sim_dir.rename(dst)


def _allocate_test_counts(
    *,
    strata_sizes: dict[tuple[str, str], int],
    test_ratio: float,
    target_test_total: int,
) -> dict[tuple[str, str], int]:
    # Base allocation: floor(test_ratio*n), but:
    # - for n < 2, force 0 (can't stratify)
    # - for n >= 2, ensure at least 1 test to keep representation
    base_counts: dict[tuple[str, str], int] = {}
    fractional: dict[tuple[str, str], float] = {}
    for key, n in strata_sizes.items():
        if n < 2:
            base_counts[key] = 0
            fractional[key] = 0.0
            continue
        ideal = test_ratio * n
        base = int(ideal)
        frac = float(ideal - base)
        if base == 0:
            base = 1
            frac = 0.0
        # never take all samples
        base = min(base, n - 1)
        base_counts[key] = base
        fractional[key] = frac

    current_total = sum(base_counts.values())

    def can_add(k: tuple[str, str]) -> bool:
        return base_counts[k] < max(0, strata_sizes[k] - 1)

    def can_remove(k: tuple[str, str]) -> bool:
        return base_counts[k] > 0 and strata_sizes[k] >= 2

    if current_total < target_test_total:
        need = target_test_total - current_total
        candidates = sorted(
            strata_sizes.keys(),
            key=lambda k: (fractional.get(k, 0.0), strata_sizes[k]),
            reverse=True,
        )
        for key in candidates:
            if need <= 0:
                break
            if not can_add(key):
                continue
            base_counts[key] += 1
            need -= 1

    elif current_total > target_test_total:
        need = current_total - target_test_total
        candidates = sorted(
            strata_sizes.keys(),
            key=lambda k: (fractional.get(k, 0.0), strata_sizes[k]),
        )
        for key in candidates:
            if need <= 0:
                break
            if not can_remove(key):
                continue
            # if possible, keep at least one test sample per stratum
            if base_counts[key] <= 1:
                continue
            base_counts[key] -= 1
            need -= 1

        # if still need to remove, allow dropping whole strata from test (rare)
        if need > 0:
            for key in candidates:
                if need <= 0:
                    break
                if not can_remove(key):
                    continue
                if base_counts[key] <= 0:
                    continue
                base_counts[key] -= 1
                need -= 1

    return base_counts


def stratified_train_test_split(
    simulations: list[SimulationNomenclatureRow],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[SimulationNomenclatureRow], list[SimulationNomenclatureRow]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    if not simulations:
        return [], []

    rng = random.Random(seed)
    sims_sorted = sorted(simulations, key=lambda r: (r.set_name, r.plan_type, r.simulation_name))

    strata: dict[tuple[str, str], list[SimulationNomenclatureRow]] = {}
    for row in sims_sorted:
        key = (row.set_name, row.plan_type)
        strata.setdefault(key, []).append(row)

    for key in strata:
        rng.shuffle(strata[key])

    n_total = len(sims_sorted)
    target_test_total = int(round(test_ratio * n_total))
    strata_sizes = {k: len(v) for k, v in strata.items()}
    test_counts = _allocate_test_counts(
        strata_sizes=strata_sizes,
        test_ratio=test_ratio,
        target_test_total=target_test_total,
    )

    test: list[SimulationNomenclatureRow] = []
    train: list[SimulationNomenclatureRow] = []
    for key, items in strata.items():
        n_test = test_counts.get(key, 0)
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    train = sorted(train, key=lambda r: r.simulation_name)
    test = sorted(test, key=lambda r: r.simulation_name)
    return train, test


def _write_split_reports(
    *,
    rows: list[SimulationNomenclatureRow],
    train: list[SimulationNomenclatureRow],
    test: list[SimulationNomenclatureRow],
    out_csv: Path,
    out_json: Path,
    seed: int,
    test_ratio: float,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    train_set = {r.simulation_name for r in train}
    test_set = {r.simulation_name for r in test}

    def split_label(name: str) -> str:
        if name in test_set:
            return "test"
        if name in train_set:
            return "train"
        return "unknown"

    # CSV
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["simulation_name", "global_id", "set", "plan_type", "case_id", "split"],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.simulation_name):
            writer.writerow(
                {
                    "simulation_name": row.simulation_name,
                    "global_id": row.global_id,
                    "set": row.set_name,
                    "plan_type": row.plan_type,
                    "case_id": row.case_id if row.case_id is not None else "",
                    "split": split_label(row.simulation_name),
                }
            )

    # JSON summary
    def summarize(items: list[SimulationNomenclatureRow]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in items:
            key = f"{r.set_name} | {r.plan_type}"
            counts[key] = counts.get(key, 0) + 1
        return {"count": len(items), "by_stratum": dict(sorted(counts.items(), key=lambda kv: kv[0]))}

    summary = {
        "seed": seed,
        "test_ratio": test_ratio,
        "total": summarize(rows),
        "train": summarize(train),
        "test": summarize(test),
        "train_simulations": [r.simulation_name for r in train],
        "test_simulations": [r.simulation_name for r in test],
        "out_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _existing_simulations_from_raw_root(
    raw_root: Path,
    index: dict[int, SimulationNomenclatureRow],
) -> list[SimulationNomenclatureRow]:
    if not raw_root.exists() or not raw_root.is_dir():
        return []
    found: list[SimulationNomenclatureRow] = []
    for d in raw_root.iterdir():
        if not d.is_dir():
            continue
        if d.name.lower() == "v_2":
            continue
        global_id = _infer_global_id_from_dirname(d.name)
        if global_id is None:
            continue
        row = index.get(global_id)
        if row is None:
            continue
        found.append(row)
    # de-duplicate by simulation_name
    seen: set[str] = set()
    dedup: list[SimulationNomenclatureRow] = []
    for r in sorted(found, key=lambda r: r.simulation_name):
        if r.simulation_name in seen:
            continue
        seen.add(r.simulation_name)
        dedup.append(r)
    return dedup


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Rename simulation folders to canonical nomenclature (from simulaciones_nomenclatura.csv) "
            "and create an 80/20 train-test split stratified by set and plan_type."
        )
    )
    p.add_argument("--nomenclatura-csv", type=Path, default=DEFAULT_NOMENCLATURA_CSV)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    p.add_argument("--apply", action="store_true", help="Apply directory renames (default: dry-run).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    p.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    p.add_argument("--rename-report-json", type=Path, default=DEFAULT_RENAME_REPORT_JSON)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    rows = _read_nomenclature_rows(args.nomenclatura_csv)
    index = _build_index(rows)

    report: dict[str, Any] = {
        "apply": bool(args.apply),
        "nomenclatura_csv": str(args.nomenclatura_csv),
    }

    _rename_simulation_dirs(
        root=args.raw_root,
        index=index,
        apply_changes=bool(args.apply),
        report=report,
        label="raw",
    )
    _rename_simulation_dirs(
        root=args.processed_root,
        index=index,
        apply_changes=bool(args.apply),
        report=report,
        label="processed",
    )

    args.rename_report_json.parent.mkdir(parents=True, exist_ok=True)
    args.rename_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    existing = _existing_simulations_from_raw_root(args.raw_root, index)
    train, test = stratified_train_test_split(existing, test_ratio=float(args.test_ratio), seed=int(args.seed))

    _write_split_reports(
        rows=existing,
        train=train,
        test=test,
        out_csv=args.split_csv,
        out_json=args.split_json,
        seed=int(args.seed),
        test_ratio=float(args.test_ratio),
    )

    print(f"[OK] Rename report: {args.rename_report_json}")
    print(f"[OK] Split CSV: {args.split_csv}")
    print(f"[OK] Split JSON: {args.split_json}")
    print(f"[INFO] Existing simulations considered: {len(existing)} (train={len(train)}, test={len(test)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
