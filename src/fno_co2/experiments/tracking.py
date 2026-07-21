"""Abstracción de tracking para campañas (spec-004 Fase 4, §1.5).

`FileTracker` (siempre activo, cero dependencias) es la única implementación
garantizada: `metrics_history.json`/`config.json` ya los escribe
`training/loop.py`; este tracker solo consolida las rutas de artefactos
relevantes de una corrida — no duplica ni reemplaza lo que el training loop
ya escribe.

`MlflowTracker`/`WandbTracker` son adaptadores **opcionales** que solo importan
`mlflow`/`wandb` al construirse (nunca a nivel de módulo). `build_tracker()` los
intenta y degrada a `FileTracker` con un warning si el paquete no está instalado
(§1.2.7). Instalar esas dependencias requiere confirmación explícita del usuario
(`CLAUDE.md` §Dependencias) — este módulo no las instala ni las requiere.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fno_co2.experiments.reproducibility import atomic_write_json
from fno_co2.utils import get_logger

logger = get_logger(__name__)


class ExperimentTracker(ABC):
    @abstractmethod
    def log_params(self, params: dict) -> None: ...

    @abstractmethod
    def log_metrics(self, step: int, metrics: dict) -> None: ...

    @abstractmethod
    def log_artifact(self, path: Path) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


class FileTracker(ExperimentTracker):
    """Tracker siempre activo, cero dependencias nuevas. `metrics_history.json`/
    `config.json` ya son la fuente de verdad (los escribe `training/loop.py`); este
    tracker solo consolida params/rutas de artefactos en `<run_dir>/tracker_paths.json`."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._params: dict = {}
        self._artifact_paths: list[str] = []

    def log_params(self, params: dict) -> None:
        self._params.update(params)

    def log_metrics(self, step: int, metrics: dict) -> None:
        # metrics_history.json (training/loop.py) ya es la fuente de verdad; no se duplica.
        pass

    def log_artifact(self, path: Path) -> None:
        self._artifact_paths.append(str(path))

    def finish(self) -> None:
        atomic_write_json(
            self.run_dir / "tracker_paths.json",
            {"params": self._params, "artifacts": self._artifact_paths},
        )


class MlflowTracker(ExperimentTracker):
    """Adaptador opcional (§1.5). Requiere `pip install mlflow` — **confirmar con el
    usuario antes de instalar** (`CLAUDE.md` §Dependencias). Import diferido: instanciar
    sin `mlflow` instalado lanza `ModuleNotFoundError`, que `build_tracker()` atrapa para
    degradar a `FileTracker`."""

    def __init__(self, run_name: str, tracking_uri: str | None = None):
        import mlflow  # import diferido a propósito, ver docstring de la clase

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, step: int, metrics: dict) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def finish(self) -> None:
        self._mlflow.end_run()


class WandbTracker(ExperimentTracker):
    """Adaptador opcional (§1.5). Requiere `pip install wandb` y **publica los datos de
    entrenamiento a un servicio externo** — no activar sin consentimiento explícito
    adicional (riesgos de spec-004). Import diferido igual que `MlflowTracker`."""

    def __init__(self, run_name: str, project: str | None = None):
        import wandb  # import diferido a propósito, ver docstring de la clase

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name)

    def log_params(self, params: dict) -> None:
        self._wandb.config.update(params)

    def log_metrics(self, step: int, metrics: dict) -> None:
        self._wandb.log(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        self._wandb.save(str(path))

    def finish(self) -> None:
        self._run.finish()


_OPTIONAL_BACKENDS = {"mlflow": MlflowTracker, "wandb": WandbTracker}


def build_tracker(
    backend: str,
    run_dir: Path,
    *,
    run_name: str | None = None,
    tracking_uri: str | None = None,
) -> ExperimentTracker:
    """Construye el tracker según `backend` (`campaign.tracking_backend`). Degrada a
    `FileTracker` con un warning si el backend pedido no está instalado o no se reconoce
    — nunca aborta por esto (spec-004 §1.2.7)."""
    if backend == "file":
        return FileTracker(run_dir)

    tracker_cls = _OPTIONAL_BACKENDS.get(backend)
    if tracker_cls is None:
        logger.warning(f"backend de tracking desconocido '{backend}'; se usa 'file'")
        return FileTracker(run_dir)

    try:
        if tracker_cls is MlflowTracker:
            return MlflowTracker(run_name=run_name or run_dir.name, tracking_uri=tracking_uri)
        return WandbTracker(run_name=run_name or run_dir.name)
    except ModuleNotFoundError:
        logger.warning(
            f"backend de tracking '{backend}' no está instalado; se degrada a 'file' "
            "(instalarlo requiere confirmación explícita del usuario, §Dependencias de CLAUDE.md)"
        )
        return FileTracker(run_dir)
