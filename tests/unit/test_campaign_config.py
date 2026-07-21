import sys
import types
from dataclasses import asdict

import pytest
import yaml

from fno_co2.config import Config
from fno_co2.experiments.campaign_config import (
    BASELINE_NAME,
    MIN_SEEDS,
    load_campaign_from_yaml,
    run_preflight,
)


def _write_config_yaml(path, **overrides):
    cfg_dict = asdict(Config())
    cfg_dict.update(overrides)
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")


def _write_data_dirs(data_root, train_dir="train", val_dir="test"):
    (data_root / train_dir).mkdir(parents=True, exist_ok=True)
    (data_root / train_dir / "placeholder.pt").write_text("x")
    (data_root / val_dir).mkdir(parents=True, exist_ok=True)
    (data_root / val_dir / "placeholder.pt").write_text("x")


def _register_fake_variant(name: str):
    module_name = f"fno_co2.models.variants.{name}"
    fake_module = types.ModuleType(module_name)
    fake_module.build = lambda cfg: object()
    sys.modules[module_name] = fake_module
    return module_name


def _write_split_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sim_id,split\n001,train\n002,test\n", encoding="utf-8")


def _write_campaign_yaml(path, **overrides):
    data = {
        "campaign_name": "test_campaign",
        "description": "campaña de prueba",
        "seeds": [42, 43, 44],
        "variants": [],
        "tracking": {"backend": "file"},
    }
    data.update(overrides)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def valid_setup(tmp_path):
    """Campaña mínima válida: baseline + una variante fake registrada, con datos y
    split de prueba. Devuelve (campaign_yaml_path, tmp_path) listo para preflight."""
    data_root = tmp_path / "data" / "processed"
    _write_data_dirs(data_root)

    baseline_cfg = tmp_path / "baseline.yaml"
    _write_config_yaml(
        baseline_cfg,
        data_root=str(data_root), model_variant="fno_baseline", experiment_name="baseline",
    )

    fake_variant_name = "_test_campaign_fake_variant"
    _register_fake_variant(fake_variant_name)

    fake_cfg = tmp_path / "fake_variant.yaml"
    _write_config_yaml(
        fake_cfg,
        data_root=str(data_root), model_variant=fake_variant_name, experiment_name=fake_variant_name,
    )

    split_csv = tmp_path / "split.csv"
    _write_split_csv(split_csv)

    campaign_yaml = tmp_path / "campaign.yaml"
    _write_campaign_yaml(
        campaign_yaml,
        variants=[
            {
                "name": BASELINE_NAME,
                "config": str(baseline_cfg),
                "success_criterion": "referencia (linea base)",
            },
            {
                "name": fake_variant_name,
                "config": str(fake_cfg),
                "success_criterion": {"metric": "val_sf_r2", "op": ">=", "threshold": 0.9},
            },
        ],
    )

    yield campaign_yaml, split_csv, fake_variant_name

    del sys.modules[f"fno_co2.models.variants.{fake_variant_name}"]


def test_valid_campaign_loads_and_expands_expected_queue(valid_setup):
    campaign_yaml, _split_csv, fake_variant_name = valid_setup
    campaign = load_campaign_from_yaml(campaign_yaml)

    assert campaign.campaign_name == "test_campaign"
    assert campaign.seeds == [42, 43, 44]
    assert [v.name for v in campaign.variants] == [BASELINE_NAME, fake_variant_name]

    queue = campaign.job_queue()
    assert queue == [
        (BASELINE_NAME, 42), (BASELINE_NAME, 43), (BASELINE_NAME, 44),
        (fake_variant_name, 42), (fake_variant_name, 43), (fake_variant_name, 44),
    ]


def test_valid_campaign_passes_preflight_without_errors(valid_setup):
    campaign_yaml, split_csv, _fake_variant_name = valid_setup
    campaign = load_campaign_from_yaml(campaign_yaml)

    result = run_preflight(campaign, split_path=split_csv)

    assert result.ok, result.errors
    assert result.split_checksum is not None
    assert result.effective_tracking_backend == "file"


def test_n_seeds_derives_deterministic_seeds(tmp_path):
    campaign_yaml = tmp_path / "campaign.yaml"
    _write_campaign_yaml(
        campaign_yaml, seeds=None, n_seeds=3,
        variants=[{"name": "baseline", "config": "configs/experiments/baseline.yaml"}],
    )
    campaign = load_campaign_from_yaml(campaign_yaml)
    assert campaign.seeds == [42, 43, 44]


def test_fewer_than_min_seeds_fails_preflight(valid_setup):
    campaign_yaml, split_csv, _fake_variant_name = valid_setup
    campaign = load_campaign_from_yaml(campaign_yaml)
    campaign.seeds = [42]

    result = run_preflight(campaign, split_path=split_csv)

    assert not result.ok
    assert any(f">= {MIN_SEEDS}" in err for err in result.errors)


def test_missing_success_criterion_fails_preflight_for_non_baseline(valid_setup):
    campaign_yaml, split_csv, fake_variant_name = valid_setup
    campaign = load_campaign_from_yaml(campaign_yaml)
    for variant in campaign.variants:
        if variant.name == fake_variant_name:
            variant.success_criterion = None

    result = run_preflight(campaign, split_path=split_csv)

    assert not result.ok
    assert any("success_criterion" in err for err in result.errors)


def test_baseline_is_exempt_from_success_criterion(valid_setup):
    campaign_yaml, split_csv, _fake_variant_name = valid_setup
    campaign = load_campaign_from_yaml(campaign_yaml)
    for variant in campaign.variants:
        if variant.name == BASELINE_NAME:
            variant.success_criterion = None

    result = run_preflight(campaign, split_path=split_csv)

    assert not any("success_criterion" in err for err in result.errors)


def test_nonexistent_config_fails_preflight(valid_setup, tmp_path):
    campaign_yaml, split_csv, _fake_variant_name = valid_setup
    campaign = load_campaign_from_yaml(campaign_yaml)
    campaign.variants[0].config_path = tmp_path / "no_existe.yaml"

    result = run_preflight(campaign, split_path=split_csv)

    assert not result.ok
    assert any("no existe el config" in err for err in result.errors)


def test_unregistered_variant_fails_preflight(tmp_path, valid_setup):
    campaign_yaml, split_csv, _fake_variant_name = valid_setup
    data_root = tmp_path / "data" / "processed"

    bad_cfg = tmp_path / "bad_variant.yaml"
    _write_config_yaml(
        bad_cfg,
        data_root=str(data_root), model_variant="no_existe_esta_variante", experiment_name="bad",
    )

    campaign = load_campaign_from_yaml(campaign_yaml)
    campaign.variants.append(
        type(campaign.variants[0])(
            name="bad", config_path=bad_cfg, success_criterion="criterio cualquiera",
        )
    )

    result = run_preflight(campaign, split_path=split_csv)

    assert not result.ok
    assert any("no registrada" in err for err in result.errors)


def test_missing_data_dirs_fail_preflight(tmp_path, valid_setup):
    campaign_yaml, split_csv, _fake_variant_name = valid_setup
    empty_data_root = tmp_path / "empty_data"

    lonely_cfg = tmp_path / "lonely.yaml"
    _write_config_yaml(
        lonely_cfg,
        data_root=str(empty_data_root), model_variant="fno_baseline", experiment_name="lonely",
    )

    campaign = load_campaign_from_yaml(campaign_yaml)
    campaign.variants = [
        type(campaign.variants[0])(
            name=BASELINE_NAME, config_path=lonely_cfg, success_criterion="referencia",
        )
    ]

    result = run_preflight(campaign, split_path=split_csv)

    assert not result.ok
    assert any("datos no encontrados" in err for err in result.errors)


def test_split_checksum_mismatch_aborts_preflight(valid_setup, tmp_path):
    campaign_yaml, split_csv, _fake_variant_name = valid_setup
    campaign = load_campaign_from_yaml(campaign_yaml)

    recorded_path = tmp_path / "recorded_split.sha256"
    recorded_path.write_text("checksum-distinto-de-otra-corrida", encoding="utf-8")

    result = run_preflight(
        campaign, split_path=split_csv, recorded_split_checksum_path=recorded_path,
    )

    assert not result.ok
    assert any("split cambió" in err for err in result.errors)


def test_missing_split_file_fails_preflight(valid_setup, tmp_path):
    campaign_yaml, _split_csv, _fake_variant_name = valid_setup
    campaign = load_campaign_from_yaml(campaign_yaml)

    result = run_preflight(campaign, split_path=tmp_path / "no_existe_split.csv")

    assert not result.ok
    assert any("split de referencia" in err for err in result.errors)


def test_unavailable_tracking_backend_degrades_to_file_with_warning(valid_setup):
    campaign_yaml, split_csv, _fake_variant_name = valid_setup
    campaign = load_campaign_from_yaml(campaign_yaml)
    campaign.tracking_backend = "_paquete_de_tracking_que_no_existe"

    result = run_preflight(campaign, split_path=split_csv)

    assert result.effective_tracking_backend == "file"
    assert any("se degrada a 'file'" in w for w in result.warnings)


def test_example_campaign_yaml_loads_expected_variants_and_seeds(repo_root):
    campaign_path = repo_root / "configs" / "campaigns" / "fno_vs_unet_vs_attn.yaml"
    campaign = load_campaign_from_yaml(campaign_path)

    assert campaign.campaign_name == "fno_vs_unet_vs_attn"
    assert campaign.seeds == [42, 43, 44]
    assert [v.name for v in campaign.variants] == ["baseline", "unet_film", "fno_axial_attn"]
    assert campaign.variants[0].success_criterion  # baseline: string informativo
    assert campaign.variants[1].success_criterion["metric"] == "val_sf_r2"
    assert campaign.variants[2].success_criterion["metric"] == "val_sf_rmse"
