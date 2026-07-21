#!/usr/bin/env python
"""Agregación y reporte cross-arquitectura de una campaña completa (spec-004 Fase 5).

Reutiliza `scripts/aggregate_experiments.py` (spec-001 Fase 5) sin modificarlo: itera
todas las variantes de la campaña, hace *upsert* de cada una en `docs/experiments.md`
(mecanismo existente), y genera un `campaign_report.md` consolidado con la evaluación
mecánica del `success_criterion` estructurado de cada variante.
"""
import argparse
import importlib.util
from pathlib import Path

from fno_co2.experiments.campaign_config import load_campaign_from_yaml
from fno_co2.experiments.campaign_report import write_campaign_report
from fno_co2.utils import get_logger

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_aggregate_experiments_module():
    """`scripts/` no es un paquete; se carga por ruta de archivo (mismo patrón que
    `run_campaign.py` para `run_experiment.py`)."""
    spec = importlib.util.spec_from_file_location(
        "_aggregate_experiments_module_for_campaign", SCRIPTS_DIR / "aggregate_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser():
    p = argparse.ArgumentParser(
        description="Agrega y reporta todas las variantes de una campaña vs. la línea base"
    )
    p.add_argument("--config", required=True, help="YAML de campaña (configs/campaigns/<name>.yaml)")
    p.add_argument("--outputs-root", default="outputs", help="Raíz de salida (default: outputs/)")
    p.add_argument("--docs-path", default="docs/experiments.md", help="Registro a actualizar (upsert)")
    return p


def main():
    args = build_parser().parse_args()
    campaign = load_campaign_from_yaml(args.config)
    aggregate_module = _load_aggregate_experiments_module()

    report_path = write_campaign_report(
        campaign,
        aggregate_module,
        outputs_root=Path(args.outputs_root),
        docs_path=Path(args.docs_path),
    )

    logger.info(f"Reporte de campaña escrito en {report_path}")
    logger.info(f"docs/experiments.md actualizado ({args.docs_path})")


if __name__ == "__main__":
    main()
