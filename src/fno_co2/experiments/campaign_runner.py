"""Runner de campaña: cola secuencial, resume por seed, aislamiento de fallos, estado
atómico (spec-004 Fase 3).

`run_campaign` **reutiliza** `scripts/run_experiment.py::run_experiment` por variante (no
reimplementa el loop de seeds ni el subproceso a `train.py`): por cada variante, filtra las
seeds ya completas (marcador `run.done` con firma de corrida compatible, mismo mecanismo que
`training/checkpoint.py::build_run_signature`/`check_resume_compatibility`) y solo invoca
`run_experiment` con las seeds pendientes. `scripts/run_campaign.py` es quien decide *cómo*
invocar `run_experiment` (carga el módulo real por ruta) y pasa esa función aquí — este
módulo no sabe nada de `scripts/`, solo orquesta.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from fno_co2.config import Config, load_config_from_yaml
from fno_co2.experiments.campaign_config import CampaignConfig
from fno_co2.experiments.reproducibility import atomic_write_json
from fno_co2.experiments.tracking import build_tracker
from fno_co2.training.checkpoint import build_run_signature, check_resume_compatibility
from fno_co2.utils import get_logger

logger = get_logger(__name__)

RunExperimentFn = Callable[..., dict]


class NoResumeOutputExistsError(Exception):
    """Existen salidas de una corrida previa y no se pasó --resume (spec-004 §Acciones
    prohibidas: no sobrescribir checkpoints sin respaldo)."""

    def __init__(self, existing_dirs: list[Path]):
        self.existing_dirs = existing_dirs
        dirs = ", ".join(str(d) for d in existing_dirs)
        super().__init__(
            f"ya existen salidas en: {dirs}. Usa --resume para reanudar la campaña o "
            "limpia/respalda esos directorios manualmente antes de correr sin --resume."
        )


def _derive_train_val_paths(cfg: Config) -> tuple[Path, Path]:
    root = Path(cfg.data_root)
    return root / (cfg.train_dir or "train"), root / (cfg.val_dir or "test")


def _build_variant_signature(cfg: Config) -> dict:
    train_path, val_path = _derive_train_val_paths(cfg)
    return build_run_signature(cfg, train_path, val_path)


def _job_dir(campaign_dir: Path, variant_name: str, seed: int) -> Path:
    return campaign_dir / variant_name / f"seed_{seed}"


def _run_done_compatible(run_done_path: Path, current_signature: dict) -> tuple[bool, list[str]]:
    try:
        recorded = json.loads(run_done_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"run.done ilegible o corrupto: {exc}"]
    return check_resume_compatibility(recorded.get("run_signature"), current_signature)


def _existing_job_dirs_with_output(campaign: CampaignConfig, campaign_dir: Path) -> list[Path]:
    found = []
    for variant in campaign.variants:
        for seed in campaign.seeds:
            job_dir = _job_dir(campaign_dir, variant.name, seed)
            if job_dir.exists() and any(job_dir.iterdir()):
                found.append(job_dir)
    return found


def _init_state(campaign: CampaignConfig) -> dict:
    return {
        "campaign_name": campaign.campaign_name,
        "jobs": {
            f"{variant.name}/seed_{seed}": {
                "variant": variant.name, "seed": seed, "status": "pending",
            }
            for variant in campaign.variants for seed in campaign.seeds
        },
    }


def _set_job_status(state: dict, variant_name: str, seed: int, status: str, **extra) -> None:
    key = f"{variant_name}/seed_{seed}"
    entry = state["jobs"].setdefault(key, {"variant": variant_name, "seed": seed})
    entry["status"] = status
    entry["updated_at"] = datetime.now().isoformat()
    entry.update(extra)


# Rutas relativas al job_dir. best.pt/latest.pt viven en checkpoints/ (confirmado contra
# una corrida real, outputs/baseline/seed_42/checkpoints/best.pt) — no directo en job_dir.
_TRACKED_ARTIFACTS = ("metrics_history.json", "config.json", "checkpoints/best.pt")


def _consolidate_tracker(backend: str, job_dir: Path, cfg: Config, seed: int) -> None:
    """No duplica métricas (eso ya lo escribe `training/loop.py`); solo consolida params +
    rutas de artefactos de una corrida completa vía el `ExperimentTracker` de Fase 4."""
    tracker = build_tracker(backend, job_dir, run_name=f"{job_dir.parent.name}_seed{seed}")
    tracker.log_params({"seed": seed, "model_variant": cfg.model_variant, "lr": cfg.lr})
    for artifact_name in _TRACKED_ARTIFACTS:
        artifact_path = job_dir / artifact_name
        if artifact_path.exists():
            tracker.log_artifact(artifact_path)
    tracker.finish()


def run_campaign(
    campaign: CampaignConfig,
    run_experiment_fn: RunExperimentFn,
    *,
    outputs_root: Path = Path("outputs"),
    train_script: str = "scripts/train.py",
    resume: bool = False,
    extra_args: list[str] | None = None,
) -> dict:
    """Ejecuta la matriz `variantes x seeds` secuencialmente en 1 GPU. Retorna (y persiste
    atómicamente en `campaign_state.json`) el estado final de cada trabajo.

    Sin `--resume`, rechaza de entrada si ya hay salidas de una corrida previa (protege
    checkpoints existentes). Con `--resume`, salta por variante las seeds cuyo `run.done`
    tenga una firma de corrida compatible con la config actual; el resto (incompletas,
    `failed`, o con firma incompatible) se re-ejecutan.
    """
    extra_args = extra_args or []
    campaign_dir = outputs_root / "campaigns" / campaign.campaign_name

    if not resume:
        existing = _existing_job_dirs_with_output(campaign, campaign_dir)
        if existing:
            raise NoResumeOutputExistsError(existing)

    state = _init_state(campaign)

    for variant in campaign.variants:
        cfg = load_config_from_yaml(variant.config_path)
        signature = _build_variant_signature(cfg)

        seeds_to_run = []
        for seed in campaign.seeds:
            run_done_path = _job_dir(campaign_dir, variant.name, seed) / "run.done"
            if resume and run_done_path.exists():
                compatible, reasons = _run_done_compatible(run_done_path, signature)
                if compatible:
                    _set_job_status(state, variant.name, seed, "completed", skipped=True)
                    continue
                logger.warning(
                    f"{variant.name}/seed_{seed}: run.done con firma incompatible "
                    f"({reasons}); re-ejecutando"
                )
            seeds_to_run.append(seed)

        atomic_write_json(campaign_dir / "campaign_state.json", state)

        if not seeds_to_run:
            logger.info(f"{variant.name}: todas las seeds ya completas (resume). Se salta.")
            continue

        for seed in seeds_to_run:
            _set_job_status(state, variant.name, seed, "running")
        atomic_write_json(campaign_dir / "campaign_state.json", state)

        logger.info(f"{variant.name}: corriendo seeds {seeds_to_run}")
        manifest = run_experiment_fn(
            config_path=variant.config_path,
            seeds=seeds_to_run,
            experiment_name=f"campaigns/{campaign.campaign_name}/{variant.name}",
            extra_args=extra_args,
            train_script=train_script,
            outputs_root=outputs_root,
        )

        for seed_entry in manifest["seeds"]:
            seed = seed_entry["seed"]
            status = seed_entry["status"]
            _set_job_status(state, variant.name, seed, status)
            if status == "completed":
                job_dir = _job_dir(campaign_dir, variant.name, seed)
                atomic_write_json(
                    job_dir / "run.done",
                    {
                        "run_signature": signature,
                        "returncode": seed_entry.get("returncode"),
                        "finished_at": seed_entry.get("finished_at"),
                    },
                )
                _consolidate_tracker(campaign.tracking_backend, job_dir, cfg, seed)
            else:
                logger.error(f"{variant.name}/seed_{seed}: corrida fallida (status={status})")
        atomic_write_json(campaign_dir / "campaign_state.json", state)

    return state


def seed_existing_run(
    campaign: CampaignConfig,
    variant_name: str,
    existing_run_dirs: dict[int, Path],
    *,
    outputs_root: Path = Path("outputs"),
) -> list[int]:
    """Importa corridas ya completas **fuera de la campaña** (p. ej. una línea base ya
    entrenada y congelada como `outputs/baseline/seed_<s>/`) al layout de la campaña
    (`outputs/campaigns/<name>/<variant>/seed_<s>/`), copiando `metrics_history.json`/
    `config.json`/`checkpoints/best.pt` y escribiendo un `run.done` con la **firma de
    corrida real** de `variant.config_path` — para que `run_campaign(..., resume=True)` la
    salte sin re-entrenar. No modifica ni mueve la corrida original (solo copia); no
    sobrescribe un `job_dir` de la campaña que ya tenga contenido (usa el `--resume`
    normal para eso). Retorna las seeds efectivamente importadas."""
    variant = next((v for v in campaign.variants if v.name == variant_name), None)
    if variant is None:
        raise ValueError(f"la campaña '{campaign.campaign_name}' no tiene una variante '{variant_name}'")

    cfg = load_config_from_yaml(variant.config_path)
    signature = _build_variant_signature(cfg)
    campaign_dir = outputs_root / "campaigns" / campaign.campaign_name

    imported: list[int] = []
    for seed, existing_dir in existing_run_dirs.items():
        if seed not in campaign.seeds:
            logger.warning(f"seed {seed} no está en la campaña '{campaign.campaign_name}'; se omite")
            continue

        dest_dir = _job_dir(campaign_dir, variant_name, seed)
        if dest_dir.exists() and any(dest_dir.iterdir()):
            logger.warning(f"{variant_name}/seed_{seed}: {dest_dir} ya tiene contenido; se omite")
            continue

        metrics_path = existing_dir / "metrics_history.json"
        if not metrics_path.exists():
            logger.warning(f"{existing_dir} no tiene metrics_history.json; se omite")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in ("metrics_history.json", "config.json"):
            src = existing_dir / name
            if src.exists():
                shutil.copy2(src, dest_dir / name)
        checkpoints_src = existing_dir / "checkpoints"
        if checkpoints_src.exists():
            shutil.copytree(checkpoints_src, dest_dir / "checkpoints", dirs_exist_ok=True)

        atomic_write_json(
            dest_dir / "run.done",
            {
                "run_signature": signature,
                "returncode": 0,
                "finished_at": datetime.now().isoformat(),
                "imported_from": str(existing_dir),
            },
        )
        imported.append(seed)
        logger.info(f"{variant_name}/seed_{seed}: importado desde {existing_dir}")

    return imported
