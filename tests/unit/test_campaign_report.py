import json
from pathlib import Path

import numpy as np
import pytest

from fno_co2.experiments.campaign_config import CampaignConfig, CampaignVariant
from fno_co2.experiments.campaign_report import (
    aggregate_campaign,
    evaluate_structured_criterion,
    render_campaign_report,
    write_campaign_report,
)

METRIC_KEYS = ["val_sf_r2", "val_vd_r2", "val_sf_rmse", "val_vd_rmse"]


def _write_seed_metrics(seed_dir: Path, *, val_sf_r2, val_vd_r2, val_sf_rmse, val_vd_rmse):
    seed_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "epoch": 1, "val_loss": 0.05,
        "val_sf_r2": val_sf_r2, "val_vd_r2": val_vd_r2,
        "val_sf_rmse": val_sf_rmse, "val_vd_rmse": val_vd_rmse,
    }
    (seed_dir / "metrics_history.json").write_text(json.dumps([row]), encoding="utf-8")


# Valores craft-eados: baseline sin varianza (std=0); unet_film cumple su criterio (val_sf_r2
# mean ~0.981 >= 0.974, guard val_vd_r2 mean ~0.95 >= 0.943); fno_axial_attn NO cumple
# (val_sf_rmse mean ~0.0090 > umbral 0.00864) aunque su guard sí pasa (val_vd_r2 >= 0.9598).
BASELINE_VALUES = {
    42: dict(val_sf_r2=0.99, val_vd_r2=0.96, val_sf_rmse=0.010, val_vd_rmse=0.020),
    43: dict(val_sf_r2=0.99, val_vd_r2=0.96, val_sf_rmse=0.010, val_vd_rmse=0.020),
    44: dict(val_sf_r2=0.99, val_vd_r2=0.96, val_sf_rmse=0.010, val_vd_rmse=0.020),
}
UNET_FILM_VALUES = {
    42: dict(val_sf_r2=0.980, val_vd_r2=0.950, val_sf_rmse=0.011, val_vd_rmse=0.021),
    43: dict(val_sf_r2=0.981, val_vd_r2=0.950, val_sf_rmse=0.011, val_vd_rmse=0.021),
    44: dict(val_sf_r2=0.982, val_vd_r2=0.950, val_sf_rmse=0.011, val_vd_rmse=0.021),
}
FNO_AXIAL_ATTN_VALUES = {
    42: dict(val_sf_r2=0.993, val_vd_r2=0.965, val_sf_rmse=0.0090, val_vd_rmse=0.019),
    43: dict(val_sf_r2=0.993, val_vd_r2=0.966, val_sf_rmse=0.0089, val_vd_rmse=0.019),
    44: dict(val_sf_r2=0.993, val_vd_r2=0.964, val_sf_rmse=0.0091, val_vd_rmse=0.019),
}


@pytest.fixture
def synthetic_campaign(tmp_path):
    campaign_dir = tmp_path / "outputs" / "campaigns" / "camp_report_test"
    for variant_name, values_by_seed in (
        ("baseline", BASELINE_VALUES),
        ("unet_film", UNET_FILM_VALUES),
        ("fno_axial_attn", FNO_AXIAL_ATTN_VALUES),
    ):
        for seed, values in values_by_seed.items():
            _write_seed_metrics(campaign_dir / variant_name / f"seed_{seed}", **values)

    campaign = CampaignConfig(
        campaign_name="camp_report_test",
        description="Campaña sintética de prueba (spec-004 Fase 5).",
        seeds=[42, 43, 44],
        variants=[
            CampaignVariant(name="baseline", config_path=Path("unused.yaml"), success_criterion="referencia"),
            CampaignVariant(
                name="unet_film", config_path=Path("unused.yaml"),
                success_criterion={
                    "metric": "val_sf_r2", "op": ">=", "threshold": 0.974,
                    "guard": {"metric": "val_vd_r2", "op": ">=", "threshold": 0.9430},
                },
            ),
            CampaignVariant(
                name="fno_axial_attn", config_path=Path("unused.yaml"),
                success_criterion={
                    "metric": "val_sf_rmse", "op": "<=", "threshold": 0.00864,
                    "guard": {"metric": "val_vd_r2", "op": ">=", "threshold": 0.9598},
                },
            ),
        ],
    )
    return campaign, tmp_path / "outputs"


def test_aggregate_campaign_reproduces_manual_mean_std(synthetic_campaign, aggregate_script, tmp_path):
    campaign, outputs_root = synthetic_campaign
    results = aggregate_campaign(campaign, aggregate_script, outputs_root=outputs_root, docs_path=tmp_path / "docs.md")

    expected_mean = float(np.mean([v["val_sf_r2"] for v in UNET_FILM_VALUES.values()]))
    expected_std = float(np.std([v["val_sf_r2"] for v in UNET_FILM_VALUES.values()], ddof=1))
    got = results["unet_film"]["agg"]["aggregated"]["val_sf_r2"]
    assert got["mean"] == pytest.approx(expected_mean)
    assert got["std"] == pytest.approx(expected_std)
    assert results["unet_film"]["agg"]["n_seeds"] == 3


def test_aggregate_campaign_applies_statistical_test_vs_baseline(synthetic_campaign, aggregate_script, tmp_path):
    campaign, outputs_root = synthetic_campaign
    results = aggregate_campaign(campaign, aggregate_script, outputs_root=outputs_root, docs_path=tmp_path / "docs.md")

    assert results["baseline"]["comparison"] is None
    comparison = results["unet_film"]["comparison"]
    assert comparison is not None
    for key in METRIC_KEYS:
        assert comparison[key]["test"] in ("wilcoxon", "mannwhitneyu")
        assert "effect_size" in comparison[key]
        assert "pvalue" in comparison[key]


def test_aggregate_campaign_marks_verdicts_correctly(synthetic_campaign, aggregate_script, tmp_path):
    campaign, outputs_root = synthetic_campaign
    results = aggregate_campaign(campaign, aggregate_script, outputs_root=outputs_root, docs_path=tmp_path / "docs.md")

    assert results["baseline"]["verdict"] == "N/A — es la línea base"
    assert results["unet_film"]["verdict"] == "cumplido"
    assert results["fno_axial_attn"]["verdict"].startswith("no cumplido")
    assert "val_sf_rmse" in results["fno_axial_attn"]["verdict"]


def test_aggregate_campaign_upserts_docs_for_every_variant(synthetic_campaign, aggregate_script, tmp_path):
    campaign, outputs_root = synthetic_campaign
    docs_path = tmp_path / "docs.md"
    aggregate_campaign(campaign, aggregate_script, outputs_root=outputs_root, docs_path=docs_path)

    content = docs_path.read_text(encoding="utf-8")
    for variant_name in ("baseline", "unet_film", "fno_axial_attn"):
        assert f"<!-- experiment: {variant_name} -->" in content
        assert f"<!-- /experiment: {variant_name} -->" in content


def test_evaluate_structured_criterion_inconclusive_with_few_seeds():
    agg = {"n_seeds": 2, "aggregated": {"val_sf_r2": {"mean": 0.99, "std": 0.0}}}
    verdict = evaluate_structured_criterion({"metric": "val_sf_r2", "op": ">=", "threshold": 0.9}, agg)
    assert verdict.startswith("inconcluso")


def test_evaluate_structured_criterion_na_for_free_text():
    agg = {"n_seeds": 3, "aggregated": {}}
    assert evaluate_structured_criterion("texto libre cualquiera", agg) == "N/A (línea base o criterio sin estructurar)"


def test_evaluate_structured_criterion_reports_guard_failure():
    agg = {
        "n_seeds": 3,
        "aggregated": {"val_sf_r2": {"mean": 0.98, "std": 0.0}, "val_vd_r2": {"mean": 0.90, "std": 0.0}},
    }
    criterion = {
        "metric": "val_sf_r2", "op": ">=", "threshold": 0.974,
        "guard": {"metric": "val_vd_r2", "op": ">=", "threshold": 0.943},
    }
    verdict = evaluate_structured_criterion(criterion, agg)
    assert verdict.startswith("no cumplido")
    assert "guard val_vd_r2" in verdict


def test_render_campaign_report_includes_all_variants_and_verdicts(synthetic_campaign, aggregate_script, tmp_path):
    campaign, outputs_root = synthetic_campaign
    results = aggregate_campaign(campaign, aggregate_script, outputs_root=outputs_root, docs_path=tmp_path / "docs.md")

    report = render_campaign_report(campaign, results, aggregate_script)

    assert "baseline" in report
    assert "unet_film" in report
    assert "fno_axial_attn" in report
    assert "cumplido" in report


def test_render_campaign_report_includes_reproducibility_when_manifest_exists(
    synthetic_campaign, aggregate_script, tmp_path,
):
    campaign, outputs_root = synthetic_campaign
    campaign_dir = outputs_root / "campaigns" / campaign.campaign_name
    manifest = {
        "git": {"commit_hash": "abc1234", "is_dirty": False},
        "split_checksum": "deadbeef",
        "reproducibility_dir": str(campaign_dir / "reproducibility"),
    }
    (campaign_dir / "campaign_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    results = aggregate_campaign(campaign, aggregate_script, outputs_root=outputs_root, docs_path=tmp_path / "docs.md")
    report = render_campaign_report(campaign, results, aggregate_script, campaign_dir=campaign_dir)

    assert "abc1234" in report
    assert "deadbeef" in report


def test_write_campaign_report_writes_file_to_disk(synthetic_campaign, aggregate_script, tmp_path):
    campaign, outputs_root = synthetic_campaign
    report_path = write_campaign_report(
        campaign, aggregate_script, outputs_root=outputs_root, docs_path=tmp_path / "docs.md",
    )

    assert report_path == outputs_root / "campaigns" / "camp_report_test" / "campaign_report.md"
    assert report_path.exists()
    assert "camp_report_test" in report_path.read_text(encoding="utf-8")
