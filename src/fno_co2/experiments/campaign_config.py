"""Esquema de campaña + preflight (spec-004 Fase 1).

Una campaña declara una matriz `variantes x seeds` en un YAML autocontenido
(`configs/campaigns/<name>.yaml`) que referencia configs de `spec-001` Fase 2
(`configs/experiments/<v>.yaml`). El preflight (`run_preflight`) valida todo lo
necesario ANTES de gastar GPU: configs cargables, variantes registradas,
seeds/criterios suficientes, datos presentes, checksum del split estable, y
degrada (no aborta) por GPU/disco/tracking ausentes.
"""
from __future__ import annotations

import hashlib
import importlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from fno_co2.config import Config, load_config_from_yaml
from fno_co2.models.registry import BASELINE_VARIANT

# Mismo umbral que `spec-001` Fase 6 / `aggregate_experiments.py::MIN_SEEDS_FOR_VERDICT`;
# constante separada porque vive en un módulo distinto (preflight vs. veredicto).
MIN_SEEDS = 3
# Misma derivación que `scripts/run_experiment.py::parse_seeds` (n_seeds -> 42, 43, ...).
BASE_SEED = 42
# Nombre de variante reservado para la línea base (convención ya usada por
# `aggregate_experiments.py --baseline`, default "baseline"): es la única exenta de
# `success_criterion` obligatorio.
BASELINE_NAME = "baseline"
# Estimación conservadora de espacio por corrida (pesos + estado de Adam en fp32); solo
# para avisar, no para bloquear (spec-004 §1.2.6 dice "estimado y avisado").
CHECKPOINT_SIZE_ESTIMATE_BYTES = 500 * 1024 * 1024


# ==========================================
# Esquema de campaña
# ==========================================
@dataclass
class CampaignVariant:
    name: str
    config_path: Path
    success_criterion: object | None = None  # str o dict estructurado (spec-004 §1.1)


@dataclass
class CampaignConfig:
    campaign_name: str
    description: str
    seeds: list[int]
    variants: list[CampaignVariant]
    tracking_backend: str = "file"
    mlflow_tracking_uri: str | None = None
    epochs_override: int | None = None
    source_path: Path | None = None

    def job_queue(self) -> list[tuple[str, int]]:
        return [(variant.name, seed) for variant in self.variants for seed in self.seeds]


def _resolve_seeds(data: dict) -> list[int]:
    seeds = data.get("seeds")
    n_seeds = data.get("n_seeds")
    if seeds:
        return [int(s) for s in seeds]
    if n_seeds:
        return [BASE_SEED + i for i in range(int(n_seeds))]
    raise ValueError("la campaña debe declarar 'seeds' (lista explícita) o 'n_seeds' (entero)")


def load_campaign_from_yaml(path: str | Path) -> CampaignConfig:
    """Carga y valida estructuralmente un YAML de campaña. Errores de esquema (claves
    faltantes, sin variantes) fallan aquí explícito; validaciones de contenido (configs
    inexistentes, variantes no registradas, criterios ausentes) son del preflight
    (`run_preflight`), no de la carga."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    required = {"campaign_name", "variants"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{path}: faltan claves obligatorias: {sorted(missing)}")

    raw_variants = data.get("variants") or []
    if not raw_variants:
        raise ValueError(f"{path}: la campaña no declara ninguna variante")

    seeds = _resolve_seeds(data)

    variants = []
    for entry in raw_variants:
        if "name" not in entry or "config" not in entry:
            raise ValueError(f"{path}: cada variante requiere 'name' y 'config' ({entry})")
        variants.append(
            CampaignVariant(
                name=entry["name"],
                config_path=Path(entry["config"]),
                success_criterion=entry.get("success_criterion"),
            )
        )

    tracking = data.get("tracking") or {}

    return CampaignConfig(
        campaign_name=data["campaign_name"],
        description=data.get("description", ""),
        seeds=seeds,
        variants=variants,
        tracking_backend=tracking.get("backend", "file"),
        mlflow_tracking_uri=tracking.get("mlflow_tracking_uri"),
        epochs_override=data.get("epochs_override"),
        source_path=path,
    )


# ==========================================
# Preflight (spec-004 §1.2)
# ==========================================
@dataclass
class PreflightResult:
    queue: list[tuple[str, int]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    split_checksum: str | None = None
    effective_tracking_backend: str = "file"

    @property
    def ok(self) -> bool:
        return not self.errors


def compute_file_checksum(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_variant_importable(model_variant: str) -> str | None:
    """None si `model_variant` está despachable por `build_model` (mismo mecanismo que
    `models/registry.py`, sin construir el modelo); mensaje de error en caso contrario."""
    if model_variant == BASELINE_VARIANT:
        return None
    try:
        module = importlib.import_module(f"fno_co2.models.variants.{model_variant}")
    except ModuleNotFoundError:
        return (
            f"model_variant '{model_variant}' no registrada: no existe "
            f"fno_co2.models.variants.{model_variant} (spec-001 Fase 3)"
        )
    if getattr(module, "build", None) is None:
        return f"fno_co2.models.variants.{model_variant} no define build(cfg) -> nn.Module"
    return None


def _data_present(cfg: Config) -> bool:
    root = Path(cfg.data_root)
    for split_dir in (cfg.train_dir, cfg.val_dir):
        if not split_dir:
            return False
        candidate = root / split_dir
        if not candidate.is_dir() or not any(candidate.iterdir()):
            return False
    return True


def _first_existing_ancestor(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path(".")


def _check_gpu_warning() -> str | None:
    if not torch.cuda.is_available():
        return (
            "GPU no disponible en esta máquina (torch.cuda.is_available()=False); "
            "necesaria para la Fase 7 (ejecución real de la campaña)"
        )
    return None


def _check_disk_warning(n_jobs: int, probe_dir: Path) -> str | None:
    free_bytes = shutil.disk_usage(probe_dir).free
    estimated_needed = n_jobs * CHECKPOINT_SIZE_ESTIMATE_BYTES
    if free_bytes < estimated_needed:
        return (
            f"espacio en disco posiblemente insuficiente: libres {free_bytes / 1e9:.1f} GB, "
            f"estimado necesario ~{estimated_needed / 1e9:.1f} GB para {n_jobs} corridas "
            "(estimación conservadora de checkpoints, no exacta)"
        )
    return None


def _check_tracking_backend(backend: str) -> tuple[str, str | None]:
    """Retorna (backend_efectivo, warning|None). Degrada a 'file' si el backend pedido no
    está instalado (spec-004 §1.2.7); nunca aborta el preflight por esto."""
    if backend == "file":
        return "file", None
    try:
        importlib.import_module(backend)
        return backend, None
    except ModuleNotFoundError:
        return "file", (
            f"backend de tracking '{backend}' no está instalado; se degrada a 'file' "
            "(instalarlo requiere confirmación explícita del usuario, §Dependencias de CLAUDE.md)"
        )


def run_preflight(
    campaign: CampaignConfig,
    *,
    split_path: Path = Path("reports/train_test_split_80_20.csv"),
    recorded_split_checksum_path: Path | None = None,
) -> PreflightResult:
    result = PreflightResult(queue=campaign.job_queue())

    # 1.2.3 — seeds >= MIN_SEEDS
    if len(campaign.seeds) < MIN_SEEDS:
        result.errors.append(
            f"la campaña declara {len(campaign.seeds)} seeds; se requieren >= {MIN_SEEDS} "
            "(spec-001 Fase 6)"
        )

    for variant in campaign.variants:
        # 1.2.3 — success_criterion obligatorio salvo la línea base
        if variant.name != BASELINE_NAME and not variant.success_criterion:
            result.errors.append(
                f"variant '{variant.name}': falta success_criterion (obligatorio salvo "
                f"'{BASELINE_NAME}', spec-001 Fase 6.3)"
            )

        # 1.2.1 — config existe y carga
        if not variant.config_path.exists():
            result.errors.append(f"variant '{variant.name}': no existe el config {variant.config_path}")
            continue
        try:
            cfg = load_config_from_yaml(variant.config_path)
        except ValueError as exc:
            result.errors.append(f"variant '{variant.name}': error cargando {variant.config_path}: {exc}")
            continue

        # 1.2.2 — variante registrada en build_model (por cfg.model_variant, que es lo que
        # realmente despacha build_model; puede diferir de variant.name, ver caso 'baseline')
        variant_error = _check_variant_importable(cfg.model_variant)
        if variant_error:
            result.errors.append(f"variant '{variant.name}': {variant_error}")

        # 1.2.4 — datos presentes
        if not _data_present(cfg):
            result.errors.append(
                f"variant '{variant.name}': datos no encontrados en "
                f"{cfg.data_root}/{cfg.train_dir} o {cfg.data_root}/{cfg.val_dir}"
            )

    # 1.2.5 — checksum del split (guarda de comparabilidad)
    if not split_path.exists():
        result.errors.append(f"no existe el split de referencia: {split_path}")
    else:
        checksum = compute_file_checksum(split_path)
        result.split_checksum = checksum
        recorded_path = recorded_split_checksum_path or (
            Path("outputs/campaigns") / campaign.campaign_name / "reproducibility" / "split.sha256"
        )
        if recorded_path.exists():
            recorded = recorded_path.read_text(encoding="utf-8").strip()
            if recorded != checksum:
                result.errors.append(
                    f"el split cambió desde el inicio de esta campaña: checksum actual="
                    f"{checksum}, registrado={recorded} ({recorded_path}). Una campaña no "
                    "compara corridas con splits distintos."
                )

    # 1.2.6 — GPU y disco (avisos, no abortan)
    gpu_warning = _check_gpu_warning()
    if gpu_warning:
        result.warnings.append(gpu_warning)

    probe_dir = _first_existing_ancestor(Path("outputs/campaigns") / campaign.campaign_name)
    disk_warning = _check_disk_warning(len(result.queue), probe_dir)
    if disk_warning:
        result.warnings.append(disk_warning)

    # 1.2.7 — backend de tracking
    effective_backend, tracking_warning = _check_tracking_backend(campaign.tracking_backend)
    result.effective_tracking_backend = effective_backend
    if tracking_warning:
        result.warnings.append(tracking_warning)

    return result
