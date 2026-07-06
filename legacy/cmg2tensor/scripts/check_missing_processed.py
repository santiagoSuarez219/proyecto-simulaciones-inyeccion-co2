from __future__ import annotations

import argparse
import re
from pathlib import Path


SIM_DIR_PATTERN = re.compile(r"^(?P<id>\d{3})_")


def collect_simulation_ids(split_dir: Path) -> set[int]:
    ids: set[int] = set()
    if not split_dir.exists():
        return ids

    for entry in split_dir.iterdir():
        if not entry.is_dir():
            continue
        match = SIM_DIR_PATTERN.match(entry.name)
        if match:
            ids.add(int(match.group("id")))
    return ids


def format_ids(ids: list[int]) -> str:
    return ", ".join(f"{sim_id:03d}" for sim_id in ids) if ids else "(ninguno)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reporta qué simulaciones faltan en data/processed/train y "
            "data/processed/test respecto a un total esperado."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directorio base de processed. Default: data/processed",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=300,
        help="Cantidad total esperada de simulaciones. Default: 300",
    )
    args = parser.parse_args()

    train_ids = collect_simulation_ids(args.processed_dir / "train")
    test_ids = collect_simulation_ids(args.processed_dir / "test")
    found_ids = train_ids | test_ids

    expected_ids = set(range(1, args.expected_total + 1))
    missing_ids = sorted(expected_ids - found_ids)
    extra_ids = sorted(found_ids - expected_ids)

    print(f"Processed dir: {args.processed_dir}")
    print(f"Train encontradas: {len(train_ids)}")
    print(f"Test encontradas: {len(test_ids)}")
    print(f"Unicas encontradas: {len(found_ids)}")
    print(f"Esperadas: {args.expected_total}")
    print()
    print(f"Faltantes ({len(missing_ids)}):")
    print(format_ids(missing_ids))
    print()
    print(f"Fuera de rango ({len(extra_ids)}):")
    print(format_ids(extra_ids))


if __name__ == "__main__":
    main()
