import json

import numpy as np
import pytest


def _row(epoch, val_loss, sf_r2, vd_r2, sf_rmse, vd_rmse):
    return {
        "epoch": epoch,
        "val_loss": val_loss,
        "val_sf_r2": sf_r2,
        "val_vd_r2": vd_r2,
        "val_sf_rmse": sf_rmse,
        "val_vd_rmse": vd_rmse,
    }


def _write_metrics_history(seed_dir, rows):
    seed_dir.mkdir(parents=True, exist_ok=True)
    with open(seed_dir / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(rows, f)


def test_aggregate_experiment_reproduces_manual_mean_std(aggregate_script, tmp_path):
    exp_dir = tmp_path / "my_experiment"
    seed_r2 = {42: 0.90, 43: 0.92, 44: 0.94}
    for seed, r2 in seed_r2.items():
        _write_metrics_history(
            exp_dir / f"seed_{seed}",
            [_row(1, val_loss=0.05, sf_r2=r2, vd_r2=0.8, sf_rmse=0.02, vd_rmse=0.03)],
        )

    agg = aggregate_script.aggregate_experiment(exp_dir)
    assert agg["n_seeds"] == 3

    values = list(seed_r2.values())
    expected_mean = float(np.mean(values))
    expected_std = float(np.std(values, ddof=1))
    assert agg["aggregated"]["val_sf_r2"]["mean"] == pytest.approx(expected_mean)
    assert agg["aggregated"]["val_sf_r2"]["std"] == pytest.approx(expected_std)


def test_aggregate_picks_best_epoch_by_min_val_loss(aggregate_script, tmp_path):
    seed_dir = tmp_path / "exp" / "seed_1"
    rows = [
        _row(1, val_loss=0.10, sf_r2=0.80, vd_r2=0.70, sf_rmse=0.05, vd_rmse=0.06),
        _row(2, val_loss=0.05, sf_r2=0.95, vd_r2=0.90, sf_rmse=0.02, vd_rmse=0.03),
        _row(3, val_loss=0.08, sf_r2=0.85, vd_r2=0.75, sf_rmse=0.04, vd_rmse=0.05),
    ]
    _write_metrics_history(seed_dir, rows)

    metrics = aggregate_script.load_best_epoch_metrics(seed_dir)
    assert metrics["epoch"] == 2
    assert metrics["val_sf_r2"] == pytest.approx(0.95)


def test_compare_groups_identical_data_is_not_significant(aggregate_script):
    values = {1: 0.90, 2: 0.91, 3: 0.89}
    result = aggregate_script.compare_groups(dict(values), dict(values))
    assert result["pvalue"] > 0.05
    assert result["effect_size"] == pytest.approx(0.0)


def test_compare_groups_detects_clear_separation(aggregate_script):
    variant = {1: 0.99, 2: 0.98, 3: 0.985, 4: 0.995, 5: 0.97}
    baseline = {6: 0.50, 7: 0.52, 8: 0.48, 9: 0.51, 10: 0.49}
    result = aggregate_script.compare_groups(variant, baseline)
    assert result["pvalue"] < 0.05
    assert result["effect_size"] > 0


def test_compute_verdict_inconclusive_below_min_seeds(aggregate_script, tmp_path):
    variant_dir = tmp_path / "variant"
    baseline_dir = tmp_path / "baseline"
    for i, r2 in enumerate([0.9, 0.91]):  # solo 2 seeds, por debajo del minimo (Fase 6)
        _write_metrics_history(
            variant_dir / f"seed_{i}", [_row(1, 0.05, r2, 0.8, 0.02, 0.03)],
        )
    _write_metrics_history(baseline_dir / "seed_0", [_row(1, 0.05, 0.5, 0.5, 0.05, 0.05)])

    variant_agg = aggregate_script.aggregate_experiment(variant_dir)
    baseline_agg = aggregate_script.aggregate_experiment(baseline_dir)
    verdict = aggregate_script.compute_verdict(variant_agg, baseline_agg)
    assert "inconcluso" in verdict
    assert "n=2" in verdict
