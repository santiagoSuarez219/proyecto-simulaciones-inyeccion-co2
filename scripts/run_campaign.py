#!/usr/bin/env python
"""Orquestador de campañas de experimentos (spec-004).

Fase 1 (este entregable): solo `--dry-run` — carga el YAML de campaña, corre el
preflight (`fno_co2.experiments.campaign_config.run_preflight`) e imprime la cola
`variante x seed` sin entrenar nada. La ejecución real (cola secuencial en 1 GPU,
resume, aislamiento de fallos, gate de confirmación) se implementa en la Fase 3.
"""
import argparse
import sys
from pathlib import Path

from fno_co2.experiments.campaign_config import load_campaign_from_yaml, run_preflight
from fno_co2.utils import get_logger

logger = get_logger(__name__)


def build_parser():
    p = argparse.ArgumentParser(
        description="Orquesta una campaña de experimentos (matriz arquitectura x seeds)"
    )
    p.add_argument("--config", required=True, help="YAML de campaña (configs/campaigns/<name>.yaml)")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Solo preflight: valida e imprime la cola, no entrena (único modo disponible en Fase 1)",
    )
    p.add_argument(
        "--split-path", default="reports/train_test_split_80_20.csv",
        help="CSV del split usado para la guarda de comparabilidad (checksum)",
    )
    return p


def main():
    args = build_parser().parse_args()
    campaign = load_campaign_from_yaml(args.config)
    result = run_preflight(campaign, split_path=Path(args.split_path))

    logger.info(
        f"Campaña '{campaign.campaign_name}': {len(result.queue)} corridas en cola "
        f"({len(campaign.variants)} variantes x {len(campaign.seeds)} seeds)"
    )
    for variant_name, seed in result.queue:
        logger.info(f"  - {variant_name} / seed {seed}")

    for warning in result.warnings:
        logger.warning(warning)

    if result.errors:
        for error in result.errors:
            logger.error(error)
        logger.error(f"Preflight fallido: {len(result.errors)} error(es). No se ejecuta nada.")
        sys.exit(1)

    logger.info("Preflight OK.")

    if not args.dry_run:
        logger.error(
            "La ejecución real de la campaña (cola secuencial, resume, gate de confirmación) "
            "se implementa en la Fase 3 de spec-004. Por ahora usa --dry-run."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
