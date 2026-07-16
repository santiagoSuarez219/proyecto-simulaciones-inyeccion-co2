#!/usr/bin/env python
"""Orquestador de campañas de experimentos (spec-004).

`--dry-run`: solo preflight + cola impresa, no entrena. Sin `--dry-run`: ejecuta de verdad
la matriz `variantes x seeds`, secuencialmente en 1 GPU, reutilizando
`scripts/run_experiment.py` por variante (spec-004 Fase 3). Requiere `--yes` — gate de
confirmación explícita (`CLAUDE.md` §Despliegue): nunca lanza GPU sin consentimiento.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

from fno_co2.experiments.campaign_config import load_campaign_from_yaml, run_preflight
from fno_co2.experiments.campaign_runner import NoResumeOutputExistsError, run_campaign
from fno_co2.experiments.reproducibility import capture_reproducibility
from fno_co2.utils import get_logger

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_run_experiment_module():
    """`scripts/` no es un paquete (sin __init__.py); se carga por ruta de archivo, igual
    que el fixture `run_experiment_script` de los tests (tests/unit/conftest.py)."""
    spec = importlib.util.spec_from_file_location(
        "_run_experiment_module_for_campaign", SCRIPTS_DIR / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser():
    p = argparse.ArgumentParser(
        description="Orquesta una campaña de experimentos (matriz arquitectura x seeds)"
    )
    p.add_argument("--config", required=True, help="YAML de campaña (configs/campaigns/<name>.yaml)")
    p.add_argument("--dry-run", action="store_true", help="Solo preflight: valida e imprime la cola, no entrena")
    p.add_argument(
        "--resume", action="store_true",
        help="Reanuda: salta seeds con run.done de firma compatible; re-ejecuta failed/incompletas",
    )
    p.add_argument(
        "--yes", action="store_true",
        help="Confirma explícitamente la ejecución real (gate de confirmación, requerido sin --dry-run)",
    )
    p.add_argument(
        "--split-path", default="reports/train_test_split_80_20.csv",
        help="CSV del split usado para la guarda de comparabilidad (checksum)",
    )
    p.add_argument(
        "--extra-args", default=None,
        help="Argumentos adicionales pasados tal cual a train.py, ej: '--epochs 1'",
    )
    p.add_argument(
        "--outputs-root", default="outputs",
        help=(
            "Raíz de salida para la contabilidad de la campaña (campaign_state.json, "
            "run.done, reproducibility/). NO reubica las escrituras reales de train.py: "
            "resolve_config deriva su output_dir con el literal 'outputs/<experiment>/"
            "seed_<seed>' relativo al cwd del proceso, sin enterarse de este flag (train.py "
            "no se modifica). Dejar en 'outputs' (default) para corridas reales; solo "
            "cambiar para tests con run_experiment mockeado (sin subproceso real a train.py)."
        ),
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

    if args.dry_run:
        return

    if not args.yes:
        logger.error(
            "Ejecución real solicitada sin --yes. Por seguridad (CLAUDE.md §Despliegue), "
            "correr la campaña de verdad exige confirmación explícita: agrega --yes."
        )
        sys.exit(2)

    outputs_root = Path(args.outputs_root)
    campaign_dir = outputs_root / "campaigns" / campaign.campaign_name
    manifest_path = campaign_dir / "campaign_manifest.json"
    if manifest_path.exists():
        logger.info(f"Manifiesto de reproducibilidad ya existe ({manifest_path}); no se recaptura.")
    else:
        reproducibility_dir = capture_reproducibility(
            campaign, outputs_root=outputs_root / "campaigns", split_path=Path(args.split_path),
        )
        logger.info(f"Reproducibilidad capturada en {reproducibility_dir} (spec-004 §1.4).")

    run_experiment_module = _load_run_experiment_module()
    extra_args = args.extra_args.split() if args.extra_args else []

    try:
        run_campaign(
            campaign,
            run_experiment_module.run_experiment,
            outputs_root=outputs_root,
            train_script=str(SCRIPTS_DIR / "train.py"),
            resume=args.resume,
            extra_args=extra_args,
        )
    except NoResumeOutputExistsError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Campaña completa.")


if __name__ == "__main__":
    main()
