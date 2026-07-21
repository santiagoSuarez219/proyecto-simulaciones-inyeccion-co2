#!/usr/bin/env python
"""Agregación de resultados multi-seed y comparación estadística vs. línea base
(spec-001 Fase 5). Lee outputs/<experiment>/seed_*/metrics_history.json, agrega
mean±std sobre seeds, compara contra la línea base con un test no paramétrico y
actualiza (no sobrescribe) docs/experiments.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

METRIC_KEYS = ["val_sf_r2", "val_vd_r2", "val_sf_rmse", "val_vd_rmse"]
# Métrica primaria para el veredicto automático de "¿supera la línea base?": sf_weight
# (2.5) es el mayor peso de la loss total (ver Config), así que SF es la señal principal.
PRIMARY_METRIC = "val_sf_rmse"
MIN_SEEDS_FOR_VERDICT = 3


def seed_from_dir(seed_dir: Path) -> int:
    return int(seed_dir.name.rsplit("_", 1)[-1])


def discover_seed_dirs(experiment_dir: Path) -> list[Path]:
    return sorted((p for p in experiment_dir.glob("seed_*") if p.is_dir()), key=seed_from_dir)


def load_best_epoch_metrics(seed_dir: Path) -> dict:
    history_path = seed_dir / "metrics_history.json"
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        raise ValueError(f"metrics_history.json vacío: {history_path}")

    best_row = min(history, key=lambda row: row["val_loss"])
    return {
        "seed": seed_from_dir(seed_dir),
        "epoch": best_row["epoch"],
        **{key: float(best_row[key]) for key in METRIC_KEYS},
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def aggregate_experiment(experiment_dir: Path) -> dict:
    seed_dirs = discover_seed_dirs(experiment_dir)
    if not seed_dirs:
        raise FileNotFoundError(f"No se encontraron subdirectorios seed_* en {experiment_dir}")

    per_seed = [load_best_epoch_metrics(seed_dir) for seed_dir in seed_dirs]

    aggregated = {}
    for key in METRIC_KEYS:
        mean, std = mean_std([row[key] for row in per_seed])
        aggregated[key] = {"mean": mean, "std": std}

    return {
        "experiment_name": experiment_dir.name,
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }


def compare_groups(variant_by_seed: dict[int, float], baseline_by_seed: dict[int, float]) -> dict:
    """Wilcoxon pareado por seed si ambos grupos comparten exactamente las mismas seeds
    (comparación más potente); si no, Mann-Whitney U. El tamaño de efecto (diferencia de
    medias) y los valores crudos siempre se reportan aparte — nunca solo el p-valor."""
    variant_seeds = sorted(variant_by_seed)
    baseline_seeds = sorted(baseline_by_seed)
    variant_values = [variant_by_seed[s] for s in variant_seeds]
    baseline_values = [baseline_by_seed[s] for s in baseline_seeds]

    if variant_seeds == baseline_seeds and len(variant_seeds) >= 2:
        paired_variant = variant_values
        paired_baseline = [baseline_by_seed[s] for s in variant_seeds]
        if all(v == b for v, b in zip(paired_variant, paired_baseline)):
            statistic, pvalue = 0.0, 1.0
        else:
            statistic, pvalue = stats.wilcoxon(paired_variant, paired_baseline)
        test_name = "wilcoxon"
    elif len(variant_values) >= 1 and len(baseline_values) >= 1:
        statistic, pvalue = stats.mannwhitneyu(variant_values, baseline_values, alternative="two-sided")
        test_name = "mannwhitneyu"
    else:
        statistic, pvalue = float("nan"), float("nan")
        test_name = "n/a"

    return {
        "test": test_name,
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "effect_size": float(np.mean(variant_values) - np.mean(baseline_values)),
    }


def compute_verdict(agg: dict, baseline_agg: dict | None) -> str:
    if baseline_agg is None:
        return "N/A — es la línea base"

    if agg["n_seeds"] < MIN_SEEDS_FOR_VERDICT:
        return f"inconcluso — n={agg['n_seeds']} seeds (mínimo {MIN_SEEDS_FOR_VERDICT} requerido, spec-001 Fase 6)"

    variant_m = agg["aggregated"][PRIMARY_METRIC]
    baseline_m = baseline_agg["aggregated"][PRIMARY_METRIC]
    v_lo, v_hi = variant_m["mean"] - variant_m["std"], variant_m["mean"] + variant_m["std"]
    b_lo, b_hi = baseline_m["mean"] - baseline_m["std"], baseline_m["mean"] + baseline_m["std"]
    overlap = v_lo <= b_hi and b_lo <= v_hi

    if overlap:
        return f"inconcluso — rangos mean±std de {PRIMARY_METRIC} se solapan con la línea base"
    # val_sf_rmse: menor es mejor.
    return "supera la línea base" if variant_m["mean"] < baseline_m["mean"] else "no supera la línea base"


def render_experiment_section(agg: dict, comparison: dict | None, meta: dict) -> str:
    name = agg["experiment_name"]
    lines = [f"<!-- experiment: {name} -->", f"## {name}", ""]
    lines.append(f"- **Qué cambia vs. línea base:** {meta.get('change_description') or '(pendiente de documentar)'}")
    lines.append(f"- **Commit/rama:** {meta.get('commit_or_branch') or '(pendiente)'}")
    seeds_str = ", ".join(str(row["seed"]) for row in agg["per_seed"])
    lines.append(f"- **Seeds:** {seeds_str} (n={agg['n_seeds']})")
    lines.append(f"- **Criterio de éxito (fijado antes de correr):** {meta.get('success_criterion') or '(no registrado)'}")
    lines.append("")
    lines.append("| métrica | mean ± std | efecto vs. línea base | test | p-valor |")
    lines.append("|---|---|---|---|---|")
    for key in METRIC_KEYS:
        m = agg["aggregated"][key]
        mean_std_str = f"{m['mean']:.4f} ± {m['std']:.4f}"
        if comparison and key in comparison:
            c = comparison[key]
            row = f"| {key} | {mean_std_str} | {c['effect_size']:+.4f} | {c['test']} | {c['pvalue']:.4f} |"
        else:
            row = f"| {key} | {mean_std_str} | — | — | — |"
        lines.append(row)
    lines.append("")
    lines.append("Valores crudos por seed (época del `best.pt` de cada seed):")
    lines.append("")
    lines.append("| seed | epoch | " + " | ".join(METRIC_KEYS) + " |")
    lines.append("|---|---|" + "---|" * len(METRIC_KEYS))
    for row in agg["per_seed"]:
        vals = " | ".join(f"{row[k]:.4f}" for k in METRIC_KEYS)
        lines.append(f"| {row['seed']} | {row['epoch']} | {vals} |")
    lines.append("")
    lines.append(f"**¿Supera la línea base?** {meta['verdict']}")
    lines.append("")
    lines.append(f"**Conclusión:** {meta.get('conclusion') or '(pendiente)'}")
    lines.append("")
    lines.append(f"<!-- /experiment: {name} -->")
    return "\n".join(lines)


def upsert_experiments_doc(doc_path: Path, experiment_name: str, section_text: str) -> None:
    start_marker = f"<!-- experiment: {experiment_name} -->"
    end_marker = f"<!-- /experiment: {experiment_name} -->"

    if doc_path.exists():
        content = doc_path.read_text(encoding="utf-8")
    else:
        content = (
            "# Registro de experimentos (spec-001 Fase 5)\n\n"
            "> Generado y actualizado por `scripts/aggregate_experiments.py`. No editar a mano "
            "las secciones entre marcadores `<!-- experiment: ... -->` — se sobrescriben en la "
            "próxima corrida de ese experimento. Registro append-only: nunca se borra una fila "
            "existente, solo se agregan o actualizan.\n"
        )

    if start_marker in content and end_marker in content:
        pre = content.split(start_marker)[0]
        post = content.split(end_marker)[1]
        content = pre + section_text + post
    else:
        content = content.rstrip("\n") + "\n\n" + section_text + "\n"

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(content, encoding="utf-8")


def build_parser():
    p = argparse.ArgumentParser(description="Agrega métricas multi-seed y actualiza docs/experiments.md")
    p.add_argument("--experiment", required=True, help="Nombre del experimento (subdir de --outputs-root)")
    p.add_argument("--baseline", default="baseline", help="Nombre del experimento de línea base (default: 'baseline')")
    p.add_argument("--outputs-root", default="outputs", help="Raíz donde viven outputs/<experiment>/seed_*/")
    p.add_argument("--docs-path", default="docs/experiments.md", help="Archivo de registro a actualizar")
    p.add_argument("--change-description", default=None, help="Qué cambia esta variante vs. línea base")
    p.add_argument("--commit-or-branch", default=None, help="Commit o rama exp/<nombre> de esta corrida")
    p.add_argument("--success-criterion", default=None, help="Criterio de éxito fijado ANTES de correr (Fase 6)")
    p.add_argument("--conclusion", default=None, help="Texto libre de conclusión")
    return p


def main():
    args = build_parser().parse_args()
    outputs_root = Path(args.outputs_root)
    docs_path = Path(args.docs_path)

    agg = aggregate_experiment(outputs_root / args.experiment)

    baseline_agg = None
    comparison = None
    if args.experiment != args.baseline:
        baseline_dir = outputs_root / args.baseline
        if not baseline_dir.exists():
            raise FileNotFoundError(
                f"No existe {baseline_dir} — registra primero la fila '{args.baseline}' (Fase 0) "
                "antes de comparar una variante contra ella."
            )
        baseline_agg = aggregate_experiment(baseline_dir)
        comparison = {
            key: compare_groups(
                {row["seed"]: row[key] for row in agg["per_seed"]},
                {row["seed"]: row[key] for row in baseline_agg["per_seed"]},
            )
            for key in METRIC_KEYS
        }

    meta = {
        "change_description": args.change_description or (None if baseline_agg is not None else "(es la línea base)"),
        "commit_or_branch": args.commit_or_branch,
        "success_criterion": args.success_criterion,
        "verdict": compute_verdict(agg, baseline_agg),
        "conclusion": args.conclusion,
    }

    section = render_experiment_section(agg, comparison, meta)
    upsert_experiments_doc(docs_path, args.experiment, section)
    print(f"[aggregate_experiments] {args.experiment}: {meta['verdict']}")
    print(f"[aggregate_experiments] Actualizado {docs_path}")


if __name__ == "__main__":
    main()
