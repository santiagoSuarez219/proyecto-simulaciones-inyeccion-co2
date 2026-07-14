from dataclasses import asdict

import pytest

from fno_co2.config import Config, load_config_from_yaml


def test_baseline_yaml_matches_config_defaults(repo_root):
    baseline_yaml = repo_root / "configs" / "experiments" / "baseline.yaml"
    loaded = load_config_from_yaml(baseline_yaml)
    assert asdict(loaded) == asdict(Config())


def test_load_config_from_yaml_rejects_unknown_keys(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("not_a_real_config_field: 1\n")
    with pytest.raises(ValueError):
        load_config_from_yaml(bad_yaml)


def test_cli_flag_overrides_yaml_value(train_script, repo_root):
    baseline_yaml = repo_root / "configs" / "experiments" / "baseline.yaml"
    args = train_script.build_parser().parse_args(["--config", str(baseline_yaml), "--lr", "1e-3"])
    cfg = train_script.resolve_config(args)
    assert cfg.lr == 1e-3
    assert cfg.batch_size == Config().batch_size  # viene del YAML, sin override


def test_yaml_without_cli_override_uses_yaml_value(train_script, repo_root):
    baseline_yaml = repo_root / "configs" / "experiments" / "baseline.yaml"
    args = train_script.build_parser().parse_args(["--config", str(baseline_yaml)])
    cfg = train_script.resolve_config(args)
    assert cfg.lr == Config().lr
