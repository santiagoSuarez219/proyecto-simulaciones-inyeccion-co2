"""Agregación y reporte cross-arquitectura de una campaña (spec-004 Fase 5).

**Reutiliza** `scripts/aggregate_experiments.py::aggregate_experiment`/`compare_groups`/
`render_experiment_section`/`upsert_experiments_doc` (`spec-001` Fase 5) — no reimplementa
la estadística ni el *upsert* de `docs/experiments.md`. Lo que este módulo **añade**:

1. Evaluación **mecánica** del `success_criterion` **estructurado** (`metric`/`op`/
   `threshold` + `guard` opcional, spec-004 §1.1) — `aggregate_experiments.py` solo sabe
   evaluar el veredicto informal "¿supera la línea base?" sobre `PRIMARY_METRIC`.
2. Iteración de esa lógica sobre **todas** las variantes de una campaña de una sola vez.
3. Un `campaign_report.md` autocontenido con la tabla de todas las variantes vs. baseline
   en un solo lugar (`aggregate_experiments.py` hoy escribe una sección por variante, no
   una vista de campaña).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from fno_co2.experiments.campaign_config import BASELINE_NAME, CampaignConfig

# Mismo umbral que aggregate_experiments.py::MIN_SEEDS_FOR_VERDICT (spec-001 Fase 6);
# constante separada porque vive en un módulo distinto.
MIN_SEEDS_FOR_VERDICT = 3

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda value, threshold: value >= threshold,
    "<=": lambda value, threshold: value <= threshold,
    ">": lambda value, threshold: value > threshold,
    "<": lambda value, threshold: value < threshold,
    "==": lambda value, threshold: value == threshold,
}


def _evaluate_condition(aggregated: dict, condition: dict) -> bool:
    value = aggregated[condition["metric"]]["mean"]
    return _OPS[condition["op"]](value, condition["threshold"])


def _render_criterion_text(success_criterion: Any) -> str:
    if isinstance(success_criterion, str):
        return success_criterion
    if isinstance(success_criterion, dict):
        text = f"{success_criterion['metric']} {success_criterion['op']} {success_criterion['threshold']}"
        guard = success_criterion.get("guard")
        if guard:
            text += f" (guard: {guard['metric']} {guard['op']} {guard['threshold']})"
        return text
    return "(no registrado)"


def evaluate_structured_criterion(success_criterion: Any, agg: dict) -> str:
    """Evalúa un `success_criterion` **estructurado** (dict `{metric,op,threshold,guard?}`)
    contra las métricas agregadas de una variante. Texto libre (línea base) -> mensaje
    informativo, no un veredicto. `< MIN_SEEDS_FOR_VERDICT` seeds -> siempre "inconcluso"
    (spec-001 Fase 6), sin excepciones — nunca se declara cumplido con evidencia insuficiente."""
    if not isinstance(success_criterion, dict):
        return "N/A (línea base o criterio sin estructurar)"

    if agg["n_seeds"] < MIN_SEEDS_FOR_VERDICT:
        return f"inconcluso — n={agg['n_seeds']} seeds (mínimo {MIN_SEEDS_FOR_VERDICT} requerido)"

    aggregated = agg["aggregated"]
    main_ok = _evaluate_condition(aggregated, success_criterion)
    guard = success_criterion.get("guard")
    guard_ok = _evaluate_condition(aggregated, guard) if guard else True

    if main_ok and guard_ok:
        return "cumplido"

    reasons = []
    if not main_ok:
        reasons.append(
            f"{success_criterion['metric']} no cumple {success_criterion['op']} {success_criterion['threshold']}"
        )
    if guard and not guard_ok:
        reasons.append(f"guard {guard['metric']} no cumple {guard['op']} {guard['threshold']}")
    return "no cumplido (" + "; ".join(reasons) + ")"


def aggregate_campaign(
    campaign: CampaignConfig,
    aggregate_module,
    *,
    outputs_root: Path = Path("outputs"),
    docs_path: Path = Path("docs/experiments.md"),
) -> dict[str, dict]:
    """Agrega y evalúa **todas** las variantes de la campaña (reutilizando
    `aggregate_module.aggregate_experiment`/`compare_groups`), y hace *upsert* de la
    sección de cada una en `docs_path` (reutilizando `render_experiment_section`/
    `upsert_experiments_doc`). Retorna `{variant_name: {"agg", "comparison", "verdict"}}`."""
    campaign_dir = outputs_root / "campaigns" / campaign.campaign_name

    baseline_agg = None
    for variant in campaign.variants:
        if variant.name == BASELINE_NAME:
            baseline_agg = aggregate_module.aggregate_experiment(campaign_dir / variant.name)
            break

    results: dict[str, dict] = {}
    for variant in campaign.variants:
        if variant.name == BASELINE_NAME:
            results[variant.name] = {"agg": baseline_agg, "comparison": None, "verdict": "N/A — es la línea base"}
            continue

        agg = aggregate_module.aggregate_experiment(campaign_dir / variant.name)
        comparison = None
        if baseline_agg is not None:
            comparison = {
                key: aggregate_module.compare_groups(
                    {row["seed"]: row[key] for row in agg["per_seed"]},
                    {row["seed"]: row[key] for row in baseline_agg["per_seed"]},
                )
                for key in aggregate_module.METRIC_KEYS
            }
        verdict = evaluate_structured_criterion(variant.success_criterion, agg)
        results[variant.name] = {"agg": agg, "comparison": comparison, "verdict": verdict}

    for variant in campaign.variants:
        result = results[variant.name]
        meta = {
            "change_description": "(es la línea base)" if variant.name == BASELINE_NAME else None,
            "commit_or_branch": campaign.campaign_name,
            "success_criterion": _render_criterion_text(variant.success_criterion),
            "verdict": result["verdict"],
            "conclusion": None,
        }
        section = aggregate_module.render_experiment_section(result["agg"], result["comparison"], meta)
        aggregate_module.upsert_experiments_doc(docs_path, variant.name, section)

    return results


def _read_reproducibility_summary(campaign_dir: Path) -> dict | None:
    manifest_path = campaign_dir / "campaign_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "commit_hash": manifest.get("git", {}).get("commit_hash"),
        "is_dirty": manifest.get("git", {}).get("is_dirty"),
        "split_checksum": manifest.get("split_checksum"),
        "reproducibility_dir": manifest.get("reproducibility_dir"),
    }


def render_campaign_report(
    campaign: CampaignConfig,
    results: dict[str, dict],
    aggregate_module,
    *,
    campaign_dir: Path | None = None,
) -> str:
    lines = [f"# Reporte de campaña: {campaign.campaign_name}", ""]
    if campaign.description:
        lines.append(campaign.description.strip())
        lines.append("")

    metric_keys = aggregate_module.METRIC_KEYS

    lines.append("## Resumen")
    lines.append("")
    header = ["variante", "n_seeds", *metric_keys, "criterio predefinido", "veredicto"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for variant in campaign.variants:
        result = results[variant.name]
        agg = result["agg"]
        metric_cells = [f"{agg['aggregated'][k]['mean']:.4f} ± {agg['aggregated'][k]['std']:.4f}" for k in metric_keys]
        criterion_text = _render_criterion_text(variant.success_criterion)
        row = [variant.name, str(agg["n_seeds"]), *metric_cells, criterion_text, result["verdict"]]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    non_baseline = [v for v in campaign.variants if v.name != BASELINE_NAME and results[v.name]["comparison"]]
    if non_baseline:
        lines.append("## Comparación estadística vs. línea base")
        lines.append("")
        lines.append("| variante | métrica | efecto | test | p-valor |")
        lines.append("|---|---|---|---|---|")
        for variant in non_baseline:
            comparison = results[variant.name]["comparison"]
            for key in metric_keys:
                c = comparison[key]
                lines.append(f"| {variant.name} | {key} | {c['effect_size']:+.4f} | {c['test']} | {c['pvalue']:.4f} |")
        lines.append("")

    lines.append("## Detalle y valores crudos por seed")
    lines.append("")
    variant_names = ", ".join(v.name for v in campaign.variants)
    lines.append(f"Ver `docs/experiments.md` (secciones actualizadas por esta corrida): {variant_names}.")
    lines.append("")

    lines.append("## Reproducibilidad")
    lines.append("")
    repro = _read_reproducibility_summary(campaign_dir) if campaign_dir else None
    if repro:
        lines.append(f"- **Commit:** `{repro['commit_hash']}` (dirty: {repro['is_dirty']})")
        lines.append(f"- **Split checksum:** `{repro['split_checksum']}`")
        lines.append(f"- **Snapshots de config y entorno:** `{repro['reproducibility_dir']}`")
    else:
        lines.append("- (sin `campaign_manifest.json` — no se capturó reproducibilidad para esta campaña)")
    lines.append("")

    return "\n".join(lines)


def write_campaign_report(
    campaign: CampaignConfig,
    aggregate_module,
    *,
    outputs_root: Path = Path("outputs"),
    docs_path: Path = Path("docs/experiments.md"),
) -> Path:
    """Orquesta la Fase 5 completa: agrega+evalúa todas las variantes, actualiza
    `docs_path` por variante, y escribe `campaign_report.md` autocontenido. Retorna la
    ruta del reporte."""
    campaign_dir = outputs_root / "campaigns" / campaign.campaign_name
    results = aggregate_campaign(campaign, aggregate_module, outputs_root=outputs_root, docs_path=docs_path)
    report_text = render_campaign_report(campaign, results, aggregate_module, campaign_dir=campaign_dir)

    report_path = campaign_dir / "campaign_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return report_path
