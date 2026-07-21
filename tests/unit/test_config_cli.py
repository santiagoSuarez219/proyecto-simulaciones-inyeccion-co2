from fno_co2.config import Config


def test_seed_flag_overrides_config_default(train_script):
    args = train_script.build_parser().parse_args(["--seed", "123"])
    cfg = train_script.resolve_config(args)
    assert cfg.seed == 123


def test_seed_flag_default_matches_config_default(train_script):
    args = train_script.build_parser().parse_args([])
    cfg = train_script.resolve_config(args)
    assert cfg.seed == Config().seed


def test_model_variant_reflected_in_run_signature(train_script, tmp_path):
    from fno_co2.training.checkpoint import build_run_signature

    args = train_script.build_parser().parse_args(["--model-variant", "fno_with_attention"])
    cfg = train_script.resolve_config(args)
    sig = build_run_signature(cfg, tmp_path / "train", tmp_path / "val")
    assert sig["model_name"] == "fno_with_attention"


def test_experiment_name_derives_distinct_output_dirs_per_seed(train_script):
    args_a = train_script.build_parser().parse_args(["--experiment-name", "exp_a", "--seed", "1"])
    args_b = train_script.build_parser().parse_args(["--experiment-name", "exp_a", "--seed", "2"])
    cfg_a = train_script.resolve_config(args_a)
    cfg_b = train_script.resolve_config(args_b)

    assert cfg_a.output_dir != cfg_b.output_dir
    assert cfg_a.output_dir == "outputs/exp_a/seed_1"
    assert cfg_b.output_dir == "outputs/exp_a/seed_2"


def test_explicit_output_dir_wins_over_experiment_name_derivation(train_script):
    args = train_script.build_parser().parse_args(
        ["--experiment-name", "exp_a", "--output-dir", "custom/path"]
    )
    cfg = train_script.resolve_config(args)
    assert cfg.output_dir == "custom/path"


def test_no_experiment_name_keeps_existing_output_dir_behavior(train_script):
    args = train_script.build_parser().parse_args(["--output-dir", "outputs/legacy_run"])
    cfg = train_script.resolve_config(args)
    assert cfg.output_dir == "outputs/legacy_run"
    assert cfg.experiment_name == Config().experiment_name
