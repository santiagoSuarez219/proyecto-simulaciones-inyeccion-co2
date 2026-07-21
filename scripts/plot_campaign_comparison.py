#!/usr/bin/env python
"""CLI para generar figuras comparativas agregadas de una campaña (spec-006 Fase 1).

Descubre las variantes y seeds bajo `outputs/campaigns/<name>/`, carga sus
`metrics_history.json` y delega en
`fno_co2.visualization.plots.save_campaign_comparison_plots` (función pura, sin I/O).
No re-entrena ni recalcula métricas: solo lee los `metrics_history.json` ya producidos
por la campaña (`scripts/run_campaign.py`, spec-004).
"""
import argparse
from pathlib import Path

from fno_co2.utils import get_logger, load_json
from fno_co2.visualization.plots import save_campaign_comparison_plots

logger = get_logger(__name__)

_NON_VARIANT_DIRS = {"reproducibility", "comparison_figures"}


def seed_from_dir(seed_dir: Path) -> int:
    return int(seed_dir.name.rsplit("_", 1)[-1])


def discover_seed_dirs(variant_dir: Path) -> list[Path]:
    return sorted((p for p in variant_dir.glob("seed_*") if p.is_dir()), key=seed_from_dir)


def discover_variant_dirs(campaign_dir: Path) -> list[Path]:
    return sorted(
        p for p in campaign_dir.iterdir() if p.is_dir() and p.name not in _NON_VARIANT_DIRS and discover_seed_dirs(p)
    )


def load_variant_histories(campaign_dir: Path) -> dict:
    variant_histories = {}
    for variant_dir in discover_variant_dirs(campaign_dir):
        histories = []
        for seed_dir in discover_seed_dirs(variant_dir):
            history_path = seed_dir / "metrics_history.json"
            if not history_path.exists():
                logger.warning(f"Sin metrics_history.json en {seed_dir}, se omite")
                continue
            histories.append(load_json(history_path))
        if histories:
            variant_histories[variant_dir.name] = histories
    return variant_histories


def build_parser():
    p = argparse.ArgumentParser(
        description="Genera figuras comparativas agregadas (curvas de convergencia + "
        "barras de metricas finales) de las variantes de una campaña (spec-006 F1)"
    )
    p.add_argument("--campaign-dir", required=True, help="outputs/campaigns/<name>/")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directorio de salida (default: <campaign-dir>/comparison_figures/)",
    )
    return p


def main():
    args = build_parser().parse_args()
    campaign_dir = Path(args.campaign_dir)
    out_dir = Path(args.out_dir) if args.out_dir else campaign_dir / "comparison_figures"

    variant_histories = load_variant_histories(campaign_dir)
    if not variant_histories:
        raise FileNotFoundError(f"No se encontraron variantes con metrics_history.json bajo {campaign_dir}")

    logger.info(f"Variantes encontradas: {list(variant_histories.keys())}")
    save_campaign_comparison_plots(variant_histories, out_dir)
    logger.info(f"Figuras comparativas escritas en {out_dir}")


if __name__ == "__main__":
    main()
