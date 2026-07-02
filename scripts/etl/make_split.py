"""
make_split.py — Stratified 90/10 train/test split for simulation folders.

Usage:
    python scripts/make_split.py [--dir data/raw] [--dry-run]

What it does:
    1. Reads simulaciones_nomenclatura.csv and performs a stratified split
       (stratify by plan_type, 90% train / 10% test, seed=42).
    2. Saves simulaciones_split.csv with a 'split' column (first run only).
    3. Creates <dir>/train/ and <dir>/test/ if they don't exist.
    4. Finds simulation folders in <dir>/, <dir>/train/, and <dir>/test/,
       then moves each one to the correct subfolder per the split assignment.

Run again whenever new simulation folders are added (to any target directory).
The split assignment is deterministic (fixed seed), so existing assignments
never change.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent  # project root (one level above scripts/)
NOMENCLATURA_CSV = ROOT / "simulaciones_nomenclatura.csv"
SPLIT_CSV = ROOT / "simulaciones_split.csv"
RANDOM_SEED = 42
TEST_SIZE = 0.10


def generate_split(df: pd.DataFrame) -> pd.DataFrame:
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["plan_type"],
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df


def print_split_summary(df: pd.DataFrame) -> None:
    print("\nSplit summary:")
    summary = df.groupby(["plan_type", "split"]).size().unstack(fill_value=0)
    print(summary.to_string())
    totals = df["split"].value_counts().sort_index()
    print(f"\nTotal  train={totals.get('train', 0)}  test={totals.get('test', 0)}\n")


def find_simulation(sim_name: str, global_id: int, base_dir: Path) -> Path | None:
    """Look for sim_name (or its numeric global_id alias) in base_dir/, base_dir/train/, and base_dir/test/."""
    numeric = str(global_id)
    for name in [sim_name, numeric]:
        for parent in [base_dir, base_dir / "train", base_dir / "test"]:
            candidate = parent / name
            if candidate.exists():
                return candidate
    return None


def _is_dir_empty(path: Path) -> bool:
    try:
        next(path.iterdir())
        return False
    except StopIteration:
        return True


def _find_numeric_orphan(global_id: int, base_dir: Path) -> Path | None:
    numeric = str(int(global_id))
    for parent in [base_dir, base_dir / "train", base_dir / "test"]:
        candidate = parent / numeric
        if candidate.exists():
            return candidate
    return None


def organize_folders(df: pd.DataFrame, base_dir: Path, dry_run: bool) -> None:
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"

    if not dry_run:
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

    moved, skipped, missing, orphans_removed, rescued_empty = 0, 0, 0, 0, 0

    for _, row in df.iterrows():
        sim_name = row["simulation_name"]
        global_id = int(row["global_id"])
        split = row["split"]
        dest_dir = train_dir if split == "train" else test_dir
        dst = dest_dir / sim_name

        src = find_simulation(sim_name, global_id, base_dir)

        if src is None:
            missing += 1
            continue

        # If the canonical destination exists but is empty, prefer moving the numeric folder in its place.
        # This handles numeric "orphans" at data/raw/<id> when an empty train/<sim_name> folder exists.
        if dst.exists() and dst.is_dir() and _is_dir_empty(dst):
            numeric_src = _find_numeric_orphan(global_id, base_dir)
            if (
                numeric_src is not None
                and numeric_src.is_dir()
                and numeric_src != dst
                and (not _is_dir_empty(numeric_src))
            ):
                if dry_run:
                    print(
                        f"  [dry-run] replace empty {split}/{sim_name} with numeric {numeric_src.relative_to(base_dir)}"
                    )
                else:
                    dst.rmdir()  # safe because dst is empty
                    shutil.move(str(numeric_src), str(dst))
                    print(
                        f"  replaced empty {split}/{sim_name} with numeric {numeric_src.relative_to(base_dir)}"
                    )
                rescued_empty += 1
                continue

        if src == dst:
            skipped += 1
            continue

        # Organized copy already exists; src is a numeric orphan — remove it.
        if dst.exists() and src.name != sim_name:
            if dry_run:
                print(f"  [dry-run] remove orphan {src.relative_to(base_dir)} (already exists as {split}/{sim_name})")
            else:
                shutil.rmtree(str(src))
                print(f"  removed orphan {src.relative_to(base_dir)} (already exists as {split}/{sim_name})")
            orphans_removed += 1
            continue

        # Already in place with correct name
        if dst.exists() and src.name == sim_name:
            skipped += 1
            continue

        if dry_run:
            print(f"  [dry-run] {src.relative_to(base_dir)} -> {split}/{sim_name}")
        else:
            shutil.move(str(src), str(dst))
            print(f"  moved {src.relative_to(base_dir)} -> {split}/{sim_name}")
        moved += 1

    print(
        f"\nDone: {moved} moved, {rescued_empty} rescued_empty, {skipped} already in place, "
        f"{orphans_removed} orphans removed, {missing} not yet available."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default="data/raw",
        help="Target directory to organize (default: data/raw)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without moving anything."
    )
    args = parser.parse_args()

    base_dir = ROOT / args.dir

    df = pd.read_csv(NOMENCLATURA_CSV)

    if SPLIT_CSV.exists():
        print(f"Loading existing split from {SPLIT_CSV.name}")
        df = pd.read_csv(SPLIT_CSV)
    else:
        print("Generating new stratified split...")
        df = generate_split(df)
        if not args.dry_run:
            df.to_csv(SPLIT_CSV, index=False)
            print(f"Saved {SPLIT_CSV.name}")

    print_split_summary(df)
    print(f"Organizing: {base_dir}")
    organize_folders(df, base_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
