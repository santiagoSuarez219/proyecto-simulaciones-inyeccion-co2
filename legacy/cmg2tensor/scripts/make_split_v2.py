"""Split estratificado 80/20 train/test sobre todas las simulaciones.

Descubre simulaciones en data/processed/:
  - *_LatinHyperCube  → plan_type="latin_hypercube"
  - NNN_<suffix>      → plan_type=<suffix>  (ej. normal, high_to_low, ...)

Excluye: histogramas, Set 1.

Genera: reports/train_test_split_all.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_CSV = BASE_DIR / "reports" / "train_test_split_all.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42
_EXCLUDE = {"histogramas", "Set 1"}


def discover_simulations() -> list[dict]:
    sims: list[dict] = []

    for d in sorted(PROCESSED_DIR.iterdir()):
        if not d.is_dir() or d.name in _EXCLUDE:
            continue

        if d.name.endswith("_LatinHyperCube"):
            plan_type = "latin_hypercube"
            source = "latin_hypercube"
        elif "_" in d.name:
            plan_type = d.name.split("_", 1)[1]
            source = "set1"
        else:
            continue  # carpeta no reconocida

        sims.append(
            {
                "simulation_name": d.name,
                "source_dir": str(d.relative_to(BASE_DIR)),
                "source": source,
                "plan_type": plan_type,
                "status": "processed",
            }
        )

    return sims


def make_split(sims: list[dict]) -> list[dict]:
    indices = list(range(len(sims)))
    stratify = [s["plan_type"] for s in sims]

    train_idxs, test_idxs = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    train_set = set(train_idxs)
    for i, s in enumerate(sims):
        s["split"] = "train" if i in train_set else "test"

    return sims


def write_csv(sims: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["simulation_name", "source_dir", "source", "plan_type", "status", "split"]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sims)


def print_summary(sims: list[dict]) -> None:
    total = len(sims)
    n_train = sum(1 for s in sims if s["split"] == "train")
    n_test = sum(1 for s in sims if s["split"] == "test")
    print(f"\nTotal simulaciones : {total}")
    print(f"  Train            : {n_train}  ({n_train/total*100:.1f}%)")
    print(f"  Test             : {n_test}  ({n_test/total*100:.1f}%)")

    plan_types = sorted(set(s["plan_type"] for s in sims))
    print(f"\nDistribución por plan_type ({len(plan_types)} clases):")
    print(f"  {'plan_type':<22} {'train':>6} {'test':>6} {'total':>6}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6}")
    for pt in plan_types:
        pt_train = sum(1 for s in sims if s["plan_type"] == pt and s["split"] == "train")
        pt_test = sum(1 for s in sims if s["plan_type"] == pt and s["split"] == "test")
        pt_total = pt_train + pt_test
        print(f"  {pt:<22} {pt_train:>6} {pt_test:>6} {pt_total:>6}")

    sources = sorted(set(s["source"] for s in sims))
    print(f"\nDistribución por source:")
    for src in sources:
        s_train = sum(1 for s in sims if s["source"] == src and s["split"] == "train")
        s_test = sum(1 for s in sims if s["source"] == src and s["split"] == "test")
        print(f"  {src:<22} train={s_train}, test={s_test}")

    n_processed = sum(1 for s in sims if s["status"] == "processed")
    n_raw = sum(1 for s in sims if s["status"] == "raw")
    print(f"\nEstado de datos:")
    print(f"  Procesadas (data/processed/) : {n_processed}")
    print(f"  Sin procesar (data/raw/)     : {n_raw}")


if __name__ == "__main__":
    sims = discover_simulations()
    if not sims:
        raise SystemExit("[ERROR] No se encontraron simulaciones.")

    sims = make_split(sims)
    write_csv(sims, OUTPUT_CSV)
    print_summary(sims)
    print(f"\nCSV guardado en: {OUTPUT_CSV}")
