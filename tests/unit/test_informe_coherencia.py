"""Guardas de coherencia numérica del informe de divulgación (spec-006 Fase 3).

`docs/informe-resultados-campana-fno-vs-unet-vs-attn.md` copia a mano cifras de
`outputs/campaigns/fno_vs_unet_vs_attn/campaign_report.md` y de los `metrics_history.json`
de la campaña real. Ninguno de esos artefactos está versionado en git (`outputs/` es
gitignored salvo estructura, ver CLAUDE.md) — por eso los tests que verifican el informe
**contra la campaña real** hacen `pytest.skip` si esos archivos no están presentes (p.ej.
en un checkout limpio o en CI), mientras que el mecanismo de comparación en sí se valida de
forma portable con datos sintéticos (no depende de la campaña real).
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_KEYS = ["val_sf_r2", "val_vd_r2", "val_sf_rmse", "val_vd_rmse"]
VARIANTS = ["baseline", "unet_film", "fno_axial_attn"]

INFORME_PATH = REPO_ROOT / "docs" / "informe-resultados-campana-fno-vs-unet-vs-attn.md"
CAMPAIGN_DIR = REPO_ROOT / "outputs" / "campaigns" / "fno_vs_unet_vs_attn"
CAMPAIGN_REPORT_PATH = CAMPAIGN_DIR / "campaign_report.md"


def _parse_metrics_row(markdown_text: str, variant: str) -> dict | None:
    """Extrae las 4 celdas de `METRIC_KEYS` de la fila de `variant` en una tabla markdown
    con columnas `| variante | n_seeds | val_sf_r2 | val_vd_r2 | val_sf_rmse | val_vd_rmse | ... |`."""
    pattern = rf"^\|\s*{re.escape(variant)}\s*\|(.+)$"
    for line in markdown_text.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            cells = [c.strip() for c in match.group(1).split("|")]
            if len(cells) < 5:
                continue
            return dict(zip(METRIC_KEYS, cells[1:5]))
    return None


def _last_calibrated_uncertainty_row(history: list[dict]) -> dict | None:
    """Ultima fila con `val_sf_uncertainty_mean > 0` (spec-004-debt-001: la fila final
    puede ser 0.0 porque `uncertainty_eval_interval` no coincidio con la ultima epoca de
    early stopping). Devuelve `None` si ninguna fila del historial calibro incertidumbre
    (caso `fno_axial_attn/seed_42` y `seed_43` en la campaña real: pararon antes de la
    primera epoca multiplo de `uncertainty_eval_interval`)."""
    calibrated = [row for row in history if row.get("val_sf_uncertainty_mean", 0.0) > 0]
    return calibrated[-1] if calibrated else None


# ==========================================
# Mecanismo de comparacion (portable, datos sinteticos)
# ==========================================


def test_parse_metrics_row_extracts_correct_cells():
    table = (
        "| variante | n_seeds | val_sf_r2 | val_vd_r2 | val_sf_rmse | val_vd_rmse | criterio | veredicto |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| baseline | 3 | 0.9937 ± 0.0001 | 0.9626 ± 0.0028 | 0.0091 ± 0.0001 | 0.0201 ± 0.0007 | referencia | N/A |\n"
    )
    row = _parse_metrics_row(table, "baseline")
    assert row == {
        "val_sf_r2": "0.9937 ± 0.0001",
        "val_vd_r2": "0.9626 ± 0.0028",
        "val_sf_rmse": "0.0091 ± 0.0001",
        "val_vd_rmse": "0.0201 ± 0.0007",
    }


def test_parse_metrics_row_returns_none_for_missing_variant():
    table = "| baseline | 3 | 0.99 | 0.96 | 0.01 | 0.02 | referencia | N/A |\n"
    assert _parse_metrics_row(table, "unet_film") is None


def test_coherence_check_catches_transcription_error():
    # Simula el escenario que esta guarda existe para prevenir: alguien copia a mano una
    # cifra mal al informe. El mismo parser aplicado a las dos tablas debe detectarlo.
    source_report = "| unet_film | 3 | 0.9920 ± 0.0002 | 0.9650 ± 0.0010 | 0.0103 ± 0.0001 | 0.0195 ± 0.0003 | c | v |\n"
    informe_with_typo = "| unet_film | 3 | 0.9920 ± 0.0002 | 0.9650 ± 0.0010 | 0.0130 ± 0.0001 | 0.0195 ± 0.0003 | c | v |\n"

    source_row = _parse_metrics_row(source_report, "unet_film")
    informe_row = _parse_metrics_row(informe_with_typo, "unet_film")

    assert source_row != informe_row
    assert source_row["val_sf_rmse"] != informe_row["val_sf_rmse"]


def test_last_calibrated_uncertainty_row_skips_final_zero():
    history = [
        {"epoch": 1, "val_sf_uncertainty_mean": 0.0},
        {"epoch": 10, "val_sf_uncertainty_mean": 0.31},
        {"epoch": 11, "val_sf_uncertainty_mean": 0.0},
        {"epoch": 12, "val_sf_uncertainty_mean": 0.0},
    ]
    row = _last_calibrated_uncertainty_row(history)
    assert row is not None
    assert row["epoch"] == 10
    assert row["val_sf_uncertainty_mean"] > 0


def test_last_calibrated_uncertainty_row_none_when_never_calibrated():
    # Emula fno_axial_attn/seed_42 y seed_43: paran antes de la primera epoca calibrada.
    history = [
        {"epoch": 1, "val_sf_uncertainty_mean": 0.0},
        {"epoch": 2, "val_sf_uncertainty_mean": 0.0},
        {"epoch": 8, "val_sf_uncertainty_mean": 0.0},
    ]
    assert _last_calibrated_uncertainty_row(history) is None


def test_aggregate_recomputation_matches_hand_copied_table(aggregate_script, tmp_path):
    """Reproduce la guarda de la Fase 3.2 con datos sinteticos: recomputa mean+/-std desde
    `metrics_history.json` (epoca del best.pt) con la misma utilidad que usa
    `campaign_report.py`, renderiza una tabla y verifica que coincide con lo que un informe
    curado copiaria a mano."""
    exp_dir = tmp_path / "unet_film"
    seed_values = {42: 0.990, 43: 0.992, 44: 0.994}
    for seed, sf_r2 in seed_values.items():
        seed_dir = exp_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        row = {
            "epoch": 1, "val_loss": 0.05,
            "val_sf_r2": sf_r2, "val_vd_r2": 0.96, "val_sf_rmse": 0.01, "val_vd_rmse": 0.02,
        }
        (seed_dir / "metrics_history.json").write_text(json.dumps([row]), encoding="utf-8")

    agg = aggregate_script.aggregate_experiment(exp_dir)
    expected_mean = float(np.mean(list(seed_values.values())))
    expected_std = float(np.std(list(seed_values.values()), ddof=1))

    hand_copied_table_row = f"{expected_mean:.4f} ± {expected_std:.4f}"
    rendered = f"{agg['aggregated']['val_sf_r2']['mean']:.4f} ± {agg['aggregated']['val_sf_r2']['std']:.4f}"
    assert rendered == hand_copied_table_row


# ==========================================
# Verificacion contra la campaña real (skip si outputs/ no esta presente localmente)
# ==========================================


def _skip_if_no_real_campaign():
    if not CAMPAIGN_REPORT_PATH.exists():
        pytest.skip(
            f"{CAMPAIGN_REPORT_PATH} no existe (outputs/ es gitignored, spec-004-debt "
            "campaign no versionada) — solo verificable localmente tras correr la campaña real."
        )


def test_informe_exists_and_is_nonempty():
    assert INFORME_PATH.exists(), f"Falta {INFORME_PATH} (spec-006 Fase 2)"
    assert len(INFORME_PATH.read_text(encoding="utf-8")) > 500


def test_informe_metrics_table_matches_real_campaign_report():
    _skip_if_no_real_campaign()
    informe_text = INFORME_PATH.read_text(encoding="utf-8")
    report_text = CAMPAIGN_REPORT_PATH.read_text(encoding="utf-8")

    for variant in VARIANTS:
        informe_row = _parse_metrics_row(informe_text, variant)
        report_row = _parse_metrics_row(report_text, variant)
        assert informe_row is not None, f"Fila de '{variant}' no encontrada en el informe"
        assert report_row is not None, f"Fila de '{variant}' no encontrada en campaign_report.md"
        assert informe_row == report_row, (
            f"Divergencia en '{variant}': informe={informe_row} vs. "
            f"campaign_report.md={report_row} — el informe quedo desactualizado respecto "
            "a la campaña real."
        )


def test_informe_early_stopping_epochs_match_real_metrics_history():
    _skip_if_no_real_campaign()
    informe_text = INFORME_PATH.read_text(encoding="utf-8")

    epoch_counts = {}
    for variant in VARIANTS:
        seed_dirs = sorted((CAMPAIGN_DIR / variant).glob("seed_*"))
        epoch_counts[variant] = sorted(
            len(json.loads((d / "metrics_history.json").read_text(encoding="utf-8"))) for d in seed_dirs
        )

    # Las epocas de parada citadas en la discusion (§7) del informe deben seguir
    # coincidiendo con el historial real; si la campaña se re-corre y cambian, esta
    # guarda falla en vez de dejar el informe con una narrativa obsoleta.
    for variant, epochs in epoch_counts.items():
        joined = "/".join(str(e) for e in epochs)
        assert joined in informe_text, (
            f"Las epocas de parada reales de '{variant}' ({joined}) no aparecen citadas "
            "tal cual en el informe — la narrativa puede haber quedado desactualizada."
        )


def test_informe_uncertainty_caveat_never_cites_final_row_as_measurement():
    _skip_if_no_real_campaign()
    informe_text = INFORME_PATH.read_text(encoding="utf-8")

    n_calibrated = {}
    for variant in VARIANTS:
        seed_dirs = sorted((CAMPAIGN_DIR / variant).glob("seed_*"))
        calibrated = 0
        for seed_dir in seed_dirs:
            history = json.loads((seed_dir / "metrics_history.json").read_text(encoding="utf-8"))
            if _last_calibrated_uncertainty_row(history) is not None:
                calibrated += 1
        n_calibrated[variant] = calibrated

    # Auditoria (spec-006 §1.4 / spec-004-debt-001): fno_axial_attn solo tiene 1 semilla
    # con incertidumbre calibrada; las otras dos variantes tienen sus 3.
    assert n_calibrated["fno_axial_attn"] == 1
    assert n_calibrated["baseline"] == 3
    assert n_calibrated["unet_film"] == 3

    # El informe debe declarar explicitamente la salvedad, no callarla.
    assert "n=1" in informe_text
    assert "artefacto" in informe_text.lower()
