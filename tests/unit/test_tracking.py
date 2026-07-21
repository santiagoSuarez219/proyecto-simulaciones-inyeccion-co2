import json

import pytest

from fno_co2.experiments.tracking import FileTracker, build_tracker


def test_file_tracker_consolidates_params_and_artifacts_without_deps(tmp_path):
    tracker = build_tracker("file", tmp_path)
    assert isinstance(tracker, FileTracker)

    tracker.log_params({"lr": 8e-4, "hidden_dim": 128})
    tracker.log_artifact(tmp_path / "best.pt")
    tracker.log_artifact(tmp_path / "config.json")
    tracker.finish()

    written = json.loads((tmp_path / "tracker_paths.json").read_text(encoding="utf-8"))
    assert written["params"] == {"lr": 8e-4, "hidden_dim": 128}
    assert written["artifacts"] == [str(tmp_path / "best.pt"), str(tmp_path / "config.json")]


def test_file_tracker_log_metrics_does_not_raise_and_does_not_duplicate_history(tmp_path):
    tracker = FileTracker(tmp_path)
    tracker.log_metrics(step=1, metrics={"val_sf_r2": 0.99})
    tracker.finish()
    # metrics_history.json es responsabilidad de training/loop.py, no de FileTracker
    assert not (tmp_path / "metrics_history.json").exists()


def test_build_tracker_file_backend_returns_file_tracker(tmp_path):
    tracker = build_tracker("file", tmp_path)
    assert isinstance(tracker, FileTracker)


def test_build_tracker_degrades_to_file_when_mlflow_not_installed(tmp_path, caplog):
    # mlflow no esta instalado en este entorno (confirmado en Fase 0: backend=file, sin
    # deps nuevas) -> esto ejercita la degradacion REAL, sin mockear el import.
    with caplog.at_level("WARNING"):
        tracker = build_tracker("mlflow", tmp_path, run_name="test_run")

    assert isinstance(tracker, FileTracker)
    assert any("se degrada a 'file'" in message for message in caplog.messages)


def test_build_tracker_degrades_to_file_when_wandb_not_installed(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        tracker = build_tracker("wandb", tmp_path, run_name="test_run")

    assert isinstance(tracker, FileTracker)
    assert any("se degrada a 'file'" in message for message in caplog.messages)


def test_build_tracker_unknown_backend_degrades_to_file_with_warning(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        tracker = build_tracker("_backend_inventado", tmp_path)

    assert isinstance(tracker, FileTracker)
    assert any("desconocido" in message for message in caplog.messages)
