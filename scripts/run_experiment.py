#!/usr/bin/env python
"""Runner multi-seed para experimentos (spec-001 Fase 4).

Reutiliza scripts/train.py tal cual, como subproceso por seed, para no divergir del
camino de código ya probado por la suite de tests. Cada seed corre aislada: si una falla
(p. ej. una guarda NaN/Inf), no aborta las demás.
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).with_name("train.py")


def parse_seeds(seeds_arg: str | None, n_seeds: int | None) -> list[int]:
    if seeds_arg:
        return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]
    if n_seeds:
        base_seed = 42
        return [base_seed + i for i in range(n_seeds)]
    raise ValueError("Debes especificar --seeds (ej: 1,2,3) o --n-seeds (ej: 3)")


def build_parser():
    p = argparse.ArgumentParser(description="Corre N seeds de un experimento y registra el resultado")
    p.add_argument("--config", required=True, help="YAML de configuración del experimento (configs/experiments/<name>.yaml)")
    p.add_argument("--seeds", default=None, help="Seeds explícitas separadas por coma, ej: 1,2,3")
    p.add_argument("--n-seeds", type=int, default=None, help="Número de seeds derivadas deterministicamente (42, 43, ...)")
    p.add_argument("--experiment-name", default=None, help="Default: nombre del archivo de config sin extensión")
    p.add_argument(
        "--extra-args", default=None,
        help="Argumentos adicionales pasados tal cual a train.py, ej: '--epochs 1 --overfit-sample-idx 0'",
    )
    p.add_argument("--train-script", default=str(TRAIN_SCRIPT), help="Override de la ruta a train.py (para tests)")
    return p


def write_manifest(out_dir: Path, manifest: dict) -> None:
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def run_experiment(
    config_path: Path,
    seeds: list[int],
    experiment_name: str,
    extra_args: list[str],
    train_script: str,
    outputs_root: Path = Path("outputs"),
) -> dict:
    out_dir = outputs_root / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copiado (no solo referenciado) para que el manifest siga siendo reproducible si el
    # YAML original cambia despues de correr el experimento.
    config_copy_path = out_dir / config_path.name
    shutil.copy2(config_path, config_copy_path)

    manifest = {
        "experiment_name": experiment_name,
        "config_path": str(config_copy_path),
        "seeds": [],
    }
    write_manifest(out_dir, manifest)

    for seed in seeds:
        entry = {"seed": seed, "status": "running", "started_at": datetime.now().isoformat()}
        manifest["seeds"].append(entry)
        write_manifest(out_dir, manifest)

        cmd = [
            sys.executable, train_script,
            "--config", str(config_copy_path),
            "--seed", str(seed),
            "--experiment-name", experiment_name,
            *extra_args,
        ]
        result = subprocess.run(cmd)

        entry["finished_at"] = datetime.now().isoformat()
        entry["returncode"] = result.returncode
        entry["status"] = "completed" if result.returncode == 0 else "failed"
        write_manifest(out_dir, manifest)

    return manifest


def main():
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de config: {config_path}")

    experiment_name = args.experiment_name or config_path.stem
    seeds = parse_seeds(args.seeds, args.n_seeds)
    extra_args = args.extra_args.split() if args.extra_args else []

    manifest = run_experiment(config_path, seeds, experiment_name, extra_args, args.train_script)

    failed = [s["seed"] for s in manifest["seeds"] if s["status"] == "failed"]
    if failed:
        print(f"[run_experiment] Seeds fallidas: {failed} — ver outputs/{experiment_name}/seed_<s>/train_console.log")
    print(f"[run_experiment] Manifest: outputs/{experiment_name}/run_manifest.json")


if __name__ == "__main__":
    main()
