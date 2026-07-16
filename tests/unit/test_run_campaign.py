import json
from dataclasses import asdict

import pytest
import yaml

from fno_co2.config import Config
from fno_co2.experiments.campaign_config import CampaignConfig, CampaignVariant
from fno_co2.experiments.campaign_runner import NoResumeOutputExistsError, run_campaign

# El train.py real requiere GPU/datos reales para entrenar de verdad. En estos tests se
# reutiliza el `run_experiment` REAL (scripts/run_experiment.py, vía el fixture
# run_experiment_script) pero apuntando a este script "train.py" falso — así se prueba la
# reutilización genuina del subproceso-por-seed sin gastar GPU (spec-004 Fase 3: "con
# train.py mockeado, subproceso simulado").
FAKE_TRAIN_SCRIPT_SRC = '''
import argparse
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--config")
p.add_argument("--seed", type=int)
p.add_argument("--experiment-name")
args, _unknown = p.parse_known_args()

out_dir = Path("outputs") / args.experiment_name / f"seed_{args.seed}"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics_history.json").write_text("[]")

sys.exit(1 if args.seed == 999 else 0)
'''


def _write_config_yaml(path, **overrides):
    cfg_dict = asdict(Config())
    cfg_dict.update(overrides)
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")


@pytest.fixture
def fake_train_script(tmp_path):
    script_path = tmp_path / "fake_train.py"
    script_path.write_text(FAKE_TRAIN_SCRIPT_SRC, encoding="utf-8")
    return str(script_path)


@pytest.fixture
def two_seed_campaign(tmp_path):
    config_path = tmp_path / "variant.yaml"
    _write_config_yaml(config_path, model_variant="fno_baseline", experiment_name="variant")
    return CampaignConfig(
        campaign_name="camp",
        description="",
        seeds=[1, 2],
        variants=[
            CampaignVariant(name="baseline", config_path=config_path, success_criterion="referencia"),
        ],
    )


@pytest.fixture
def _chdir_tmp_path(tmp_path, monkeypatch):
    """train.py deriva 'outputs/<experiment_name>/seed_<seed>' relativo al cwd (hardcodeado
    en resolve_config); aislar el cwd en tmp_path evita tocar el outputs/ real del repo.
    Requerido explícitamente por los tests que corren el fake train.py (no autouse: los
    tests a nivel CLI necesitan cwd=repo_root para resolver las rutas relativas reales)."""
    monkeypatch.chdir(tmp_path)


def test_run_campaign_runs_queue_and_writes_run_done(
    run_experiment_script, fake_train_script, two_seed_campaign, tmp_path, _chdir_tmp_path,
):
    state = run_campaign(
        two_seed_campaign,
        run_experiment_script.run_experiment,
        outputs_root=tmp_path / "outputs",
        train_script=fake_train_script,
    )

    assert state["jobs"]["baseline/seed_1"]["status"] == "completed"
    assert state["jobs"]["baseline/seed_2"]["status"] == "completed"

    campaign_dir = tmp_path / "outputs" / "campaigns" / "camp"
    assert (campaign_dir / "baseline" / "seed_1" / "run.done").exists()
    assert (campaign_dir / "baseline" / "seed_2" / "run.done").exists()

    # Fase 4: cada seed completa consolida su tracker (FileTracker por defecto, sin deps)
    tracker_paths = json.loads(
        (campaign_dir / "baseline" / "seed_1" / "tracker_paths.json").read_text(encoding="utf-8")
    )
    assert tracker_paths["params"]["seed"] == 1
    assert tracker_paths["params"]["model_variant"] == "fno_baseline"
    assert any("metrics_history.json" in path for path in tracker_paths["artifacts"])

    state_on_disk = json.loads((campaign_dir / "campaign_state.json").read_text(encoding="utf-8"))
    assert state_on_disk == state


def test_failed_seed_does_not_abort_the_rest(run_experiment_script, fake_train_script, tmp_path, _chdir_tmp_path):
    config_path = tmp_path / "variant.yaml"
    _write_config_yaml(config_path, model_variant="fno_baseline", experiment_name="variant")
    campaign = CampaignConfig(
        campaign_name="camp",
        description="",
        seeds=[999, 43],  # 999 esta hardcodeado para fallar en el fake train.py
        variants=[CampaignVariant(name="baseline", config_path=config_path, success_criterion="ref")],
    )

    state = run_campaign(
        campaign, run_experiment_script.run_experiment,
        outputs_root=tmp_path / "outputs", train_script=fake_train_script,
    )

    assert state["jobs"]["baseline/seed_999"]["status"] == "failed"
    assert state["jobs"]["baseline/seed_43"]["status"] == "completed"

    campaign_dir = tmp_path / "outputs" / "campaigns" / "camp"
    assert not (campaign_dir / "baseline" / "seed_999" / "run.done").exists()
    assert (campaign_dir / "baseline" / "seed_43" / "run.done").exists()


def test_resume_skips_completed_seeds_with_compatible_signature(
    run_experiment_script, fake_train_script, two_seed_campaign, tmp_path, _chdir_tmp_path,
):
    run_campaign(
        two_seed_campaign, run_experiment_script.run_experiment,
        outputs_root=tmp_path / "outputs", train_script=fake_train_script,
    )

    calls = []
    original_run_experiment = run_experiment_script.run_experiment

    def _tracking_run_experiment(**kwargs):
        calls.append(kwargs["seeds"])
        return original_run_experiment(**kwargs)

    state = run_campaign(
        two_seed_campaign, _tracking_run_experiment,
        outputs_root=tmp_path / "outputs", train_script=fake_train_script, resume=True,
    )

    assert calls == []  # ambas seeds ya completas -> no se invoca run_experiment de nuevo
    assert state["jobs"]["baseline/seed_1"]["status"] == "completed"
    assert state["jobs"]["baseline/seed_1"].get("skipped") is True


def test_resume_reruns_failed_seed_but_skips_completed_one(
    run_experiment_script, fake_train_script, tmp_path, _chdir_tmp_path,
):
    config_path = tmp_path / "variant.yaml"
    _write_config_yaml(config_path, model_variant="fno_baseline", experiment_name="variant")
    campaign = CampaignConfig(
        campaign_name="camp",
        description="",
        seeds=[999, 43],
        variants=[CampaignVariant(name="baseline", config_path=config_path, success_criterion="ref")],
    )

    run_campaign(
        campaign, run_experiment_script.run_experiment,
        outputs_root=tmp_path / "outputs", train_script=fake_train_script,
    )

    # "arreglamos" el fake train script para que ya no falle en ninguna seed
    fixed_script = tmp_path / "fixed_train.py"
    fixed_script.write_text(FAKE_TRAIN_SCRIPT_SRC.replace("args.seed == 999", "False"), encoding="utf-8")

    state = run_campaign(
        campaign, run_experiment_script.run_experiment,
        outputs_root=tmp_path / "outputs", train_script=str(fixed_script), resume=True,
    )

    assert state["jobs"]["baseline/seed_999"]["status"] == "completed"
    assert state["jobs"]["baseline/seed_43"].get("skipped") is True


def test_resume_reruns_when_run_done_signature_incompatible(
    run_experiment_script, fake_train_script, two_seed_campaign, tmp_path, _chdir_tmp_path,
):
    run_campaign(
        two_seed_campaign, run_experiment_script.run_experiment,
        outputs_root=tmp_path / "outputs", train_script=fake_train_script,
    )

    # cambia la config (hidden_dim) DESPUES de completar -> firma ya no compatible
    _write_config_yaml(
        two_seed_campaign.variants[0].config_path,
        model_variant="fno_baseline", experiment_name="variant", hidden_dim=999,
    )

    calls = []
    original_run_experiment = run_experiment_script.run_experiment

    def _tracking_run_experiment(**kwargs):
        calls.append(sorted(kwargs["seeds"]))
        return original_run_experiment(**kwargs)

    run_campaign(
        two_seed_campaign, _tracking_run_experiment,
        outputs_root=tmp_path / "outputs", train_script=fake_train_script, resume=True,
    )

    assert calls == [[1, 2]]  # ambas se re-ejecutan por firma incompatible


def test_campaign_state_is_written_atomically_no_tmp_left_behind(
    run_experiment_script, fake_train_script, two_seed_campaign, tmp_path, _chdir_tmp_path,
):
    run_campaign(
        two_seed_campaign, run_experiment_script.run_experiment,
        outputs_root=tmp_path / "outputs", train_script=fake_train_script,
    )
    campaign_dir = tmp_path / "outputs" / "campaigns" / "camp"
    assert (campaign_dir / "campaign_state.json").exists()
    assert not (campaign_dir / "campaign_state.json.tmp").exists()


def test_without_resume_raises_if_output_already_exists(
    run_experiment_script, fake_train_script, two_seed_campaign, tmp_path, _chdir_tmp_path,
):
    run_campaign(
        two_seed_campaign, run_experiment_script.run_experiment,
        outputs_root=tmp_path / "outputs", train_script=fake_train_script,
    )

    with pytest.raises(NoResumeOutputExistsError):
        run_campaign(
            two_seed_campaign, run_experiment_script.run_experiment,
            outputs_root=tmp_path / "outputs", train_script=fake_train_script, resume=False,
        )


def test_run_campaign_script_parses_resume_and_yes_flags(run_campaign_script):
    args = run_campaign_script.build_parser().parse_args(
        ["--config", "configs/campaigns/fno_vs_unet_vs_attn.yaml", "--resume", "--yes"]
    )
    assert args.resume is True
    assert args.yes is True
    assert args.dry_run is False


def test_run_campaign_script_refuses_real_execution_without_yes(run_campaign_script, tmp_path, monkeypatch):
    """Sin --dry-run ni --yes, el script debe salir explícito (gate de confirmación) antes
    de intentar cargar/ejecutar run_experiment.py — verificado contra la campaña real del
    repo (cwd sin cambiar, para que las rutas de config relativas resuelvan)."""
    monkeypatch.setattr(
        "sys.argv",
        ["run_campaign.py", "--config", "configs/campaigns/fno_vs_unet_vs_attn.yaml"],
    )
    with pytest.raises(SystemExit) as exc_info:
        run_campaign_script.main()
    assert exc_info.value.code == 2


def test_run_campaign_script_with_yes_captures_reproducibility_and_runs(
    run_campaign_script, tmp_path, monkeypatch,
):
    """--yes bypassea el gate y ejecuta de verdad. Se reemplaza run_experiment por un
    stub (sin subproceso real a train.py) para verificar la conexión completa
    capture_reproducibility -> run_campaign sin gastar GPU ni tocar el repo real."""

    def _fake_run_experiment(**kwargs):
        return {
            "experiment_name": kwargs["experiment_name"],
            "seeds": [
                {"seed": s, "status": "completed", "returncode": 0, "finished_at": "2026-07-16T00:00:00"}
                for s in kwargs["seeds"]
            ],
        }

    fake_module = type("FakeRunExperimentModule", (), {"run_experiment": staticmethod(_fake_run_experiment)})
    monkeypatch.setattr(run_campaign_script, "_load_run_experiment_module", lambda: fake_module)

    outputs_root = tmp_path / "outputs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_campaign.py",
            "--config", "configs/campaigns/fno_vs_unet_vs_attn.yaml",
            "--yes",
            "--outputs-root", str(outputs_root),
        ],
    )

    run_campaign_script.main()

    campaign_dir = outputs_root / "campaigns" / "fno_vs_unet_vs_attn"
    assert (campaign_dir / "campaign_manifest.json").exists()
    assert (campaign_dir / "reproducibility" / "split.sha256").exists()
    assert (campaign_dir / "campaign_state.json").exists()
    assert (campaign_dir / "baseline" / "seed_42" / "run.done").exists()
