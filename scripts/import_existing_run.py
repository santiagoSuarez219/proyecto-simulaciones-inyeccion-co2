#!/usr/bin/env python
"""Importa una corrida ya entrenada (fuera de una campaña) al layout de la campaña, para que
`run_campaign.py --resume` la salte sin re-entrenar (preparación de spec-004 Fase 7).

Ejemplo (reusar `baseline-v1` ya congelado en `outputs/baseline/seed_*` dentro de la campaña
`fno_vs_unet_vs_attn`, ahorrando 3 corridas completas de re-entrenamiento):

    python scripts/import_existing_run.py \\
      --config configs/campaigns/fno_vs_unet_vs_attn.yaml \\
      --variant baseline \\
      --source-root outputs/baseline

Solo copia (no mueve) los artefactos y escribe un `run.done` con la firma de corrida real de
la config de esa variante; no sobrescribe un `job_dir` de campaña que ya tenga contenido.
"""
import argparse
from pathlib import Path

from fno_co2.experiments.campaign_config import load_campaign_from_yaml
from fno_co2.experiments.campaign_runner import seed_existing_run
from fno_co2.utils import get_logger

logger = get_logger(__name__)


def build_parser():
    p = argparse.ArgumentParser(
        description="Importa una corrida ya entrenada (fuera de una campaña) al layout de la campaña"
    )
    p.add_argument("--config", required=True, help="YAML de campaña (configs/campaigns/<name>.yaml)")
    p.add_argument("--variant", required=True, help="Nombre de la variante destino (debe existir en la campaña)")
    p.add_argument(
        "--source-root", required=True,
        help="Directorio con subcarpetas seed_<s>/ ya entrenadas (p. ej. outputs/baseline)",
    )
    p.add_argument("--outputs-root", default="outputs", help="Raíz de salida de la campaña (default: outputs/)")
    return p


def main():
    args = build_parser().parse_args()
    campaign = load_campaign_from_yaml(args.config)
    source_root = Path(args.source_root)

    existing_run_dirs = {
        seed: source_root / f"seed_{seed}"
        for seed in campaign.seeds
        if (source_root / f"seed_{seed}").exists()
    }

    if not existing_run_dirs:
        logger.error(f"No se encontraron subcarpetas seed_* de la campaña en {source_root}")
        raise SystemExit(1)

    imported = seed_existing_run(
        campaign, args.variant, existing_run_dirs, outputs_root=Path(args.outputs_root),
    )
    logger.info(f"Importadas {len(imported)} seeds para '{args.variant}': {imported}")


if __name__ == "__main__":
    main()
