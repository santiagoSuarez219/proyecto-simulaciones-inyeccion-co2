"""
fix_perm_por_reports.py

Fixes permeability_layer_cubes_report.json and porosity_layer_cubes_report.json
for every simulation in data/processed/ (except 001_LatinHyperCube, which is
the canonical source and already correct).

Changes per simulation:
  - input_path  → data/raw/001_LatinHyperCube/{variable}.txt  (actual source)
  - output_dir  → data/processed/{sim_name}/{variable}_layer_cubes
  - output_files_preview → updated paths with correct sim_name

Usage:
  python scripts/fix_perm_por_reports.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SKIP = {"001_LatinHyperCube", "histogramas", "Set 1"}

REPORTS = {
    "permeability_layer_cubes_report.json": "permeability",
    "porosity_layer_cubes_report.json": "porosity",
}


def fix_report(report: dict, sim_name: str, variable: str) -> dict:
    subdir = f"{variable}_layer_cubes"
    report["input_path"] = f"data/raw/001_LatinHyperCube/{variable}.txt"
    report["output_dir"] = f"data/processed/{sim_name}/{subdir}"
    preview = report.get("output_files_preview", [])
    if preview:
        report["output_files_preview"] = [
            f"data/processed/{sim_name}/{subdir}/layer_cube_k{i+1:03d}.pt"
            for i in range(len(preview))
        ]
    return report


def main(dry_run: bool) -> None:
    sims = sorted(
        d for d in PROCESSED_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP
    )
    updated = skipped = 0

    for sim_dir in sims:
        for report_file, variable in REPORTS.items():
            path = sim_dir / report_file
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                report = json.load(f)

            fixed = fix_report(report, sim_dir.name, variable)

            if dry_run:
                print(f"[dry-run] {sim_dir.name}/{report_file}")
                print(f"  output_dir -> {fixed['output_dir']}")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(fixed, f, indent=2)
            updated += 1

    action = "[dry-run] Would update" if dry_run else "Updated"
    print(f"\n{action} {updated} report files across {len(sims)} simulations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.dry_run)
