import matplotlib.pyplot as plt
import pytest

from fno_co2.visualization.plots import (
    _best_epoch_row,
    _epoch_mean_std,
    _percentile_ylim,
    save_campaign_comparison_plots,
)


def _row(epoch, val_loss, sf_r2=0.99, vd_r2=0.96, sf_rmse=0.01, vd_rmse=0.02):
    return {
        "epoch": epoch,
        "val_loss": val_loss,
        "val_sf_r2": sf_r2,
        "val_vd_r2": vd_r2,
        "val_sf_rmse": sf_rmse,
        "val_vd_rmse": vd_rmse,
    }


def _history(epochs, val_losses, **kwargs):
    return [_row(e, vl, **kwargs) for e, vl in zip(epochs, val_losses)]


# Longitudes distintas por seed (spec-006 F1): baseline llega a la epoca 5 con sus 3
# seeds; fno_axial_attn emula early stopping muy temprano (epocas 2 y 3), como
# fno_axial_attn/seed_42 (8 epocas) y seed_43 (7 epocas) en la campaña real — aqui
# comprimido para que el test sea rapido.
BASELINE_HISTORIES = [
    _history([1, 2, 3, 4, 5], [0.05, 0.04, 0.03, 0.025, 0.02]),
    _history([1, 2, 3, 4, 5], [0.06, 0.045, 0.03, 0.024, 0.019]),
    _history([1, 2, 3, 4, 5], [0.055, 0.042, 0.031, 0.026, 0.021]),
]
UNET_FILM_HISTORIES = [
    _history([1, 2, 3, 4], [0.07, 0.05, 0.035, 0.03]),
    _history([1, 2, 3, 4, 5], [0.065, 0.048, 0.033, 0.028, 0.027]),
    _history([1, 2, 3], [0.08, 0.05, 0.04]),
]
# Solo la primera seed llega a la epoca 3 -> n=1 en esa epoca (banda de ancho 0).
FNO_AXIAL_ATTN_HISTORIES = [
    _history([1, 2, 3], [0.09, 0.06, 0.045]),
    _history([1, 2], [0.10, 0.07]),
]


@pytest.fixture
def variant_histories():
    return {
        "baseline": BASELINE_HISTORIES,
        "unet_film": UNET_FILM_HISTORIES,
        "fno_axial_attn": FNO_AXIAL_ATTN_HISTORIES,
    }


def test_epoch_mean_std_uses_only_seeds_present_at_each_epoch():
    epochs, means, stds, ns = _epoch_mean_std(FNO_AXIAL_ATTN_HISTORIES, "val_loss")

    assert list(epochs) == [1, 2, 3]
    assert list(ns) == [2, 2, 1]
    # Epoca 3: solo la primera seed llega -> n=1 -> std=0 (banda de ancho 0, sin fallar).
    assert stds[-1] == pytest.approx(0.0)
    assert means[-1] == pytest.approx(0.045)


def test_epoch_mean_std_matches_manual_mean_for_shared_epoch():
    epochs, means, stds, ns = _epoch_mean_std(BASELINE_HISTORIES, "val_loss")

    idx = list(epochs).index(1)
    assert ns[idx] == 3
    assert means[idx] == pytest.approx((0.05 + 0.06 + 0.055) / 3)


def test_best_epoch_row_picks_minimum_val_loss():
    history = _history([1, 2, 3], [0.05, 0.01, 0.03])
    best = _best_epoch_row(history)
    assert best["epoch"] == 2
    assert best["val_loss"] == pytest.approx(0.01)


def test_percentile_ylim_clips_single_epoch_outlier():
    # Inyecta un pico de una sola epoca entre muchas (inestabilidad real observada en
    # fno_axial_attn/seed_44 ep.3 de la campaña real: ~3% de las filas totales) y
    # verifica que el rango no lo sigue ciegamente: queda cerca del grueso de los datos,
    # no estirado hasta el pico.
    epochs = list(range(1, 51))
    losses = [0.03] * 50
    losses[24] = 5.0  # 1 pico en 50 puntos (2%), por debajo del recorte p2-p98
    spiky = {"only": [_history(epochs, losses)]}

    lo, hi = _percentile_ylim(spiky, "val_loss")
    assert lo < hi
    assert hi < 5.0  # el pico queda fuera del rango de vista, no domina la escala


def test_percentile_ylim_handles_constant_values():
    constant = {"only": [_history([1, 2, 3], [0.02, 0.02, 0.02])]}
    lo, hi = _percentile_ylim(constant, "val_loss")
    assert lo < 0.02 < hi


def test_save_campaign_comparison_plots_writes_both_figures(variant_histories, tmp_path):
    save_campaign_comparison_plots(variant_histories, tmp_path)

    assert (tmp_path / "campaign_convergence_curves.png").exists()
    assert (tmp_path / "campaign_final_metrics.png").exists()


def test_save_campaign_comparison_plots_closes_all_figures(variant_histories, tmp_path):
    plt.close("all")
    save_campaign_comparison_plots(variant_histories, tmp_path)
    assert plt.get_fignums() == []


def test_save_campaign_comparison_plots_handles_seeds_of_very_different_length(tmp_path):
    # Caso extremo: una variante con una sola seed y muy pocas epocas (similar a
    # fno_axial_attn/seed_43, 7 epocas, frente a otras seeds que llegan a 25). No debe
    # fallar ni producir NaN en las figuras.
    histories = {
        "baseline": BASELINE_HISTORIES,
        "single_short_seed": [_history([1, 2], [0.05, 0.04])],
    }
    save_campaign_comparison_plots(histories, tmp_path)
    assert (tmp_path / "campaign_convergence_curves.png").exists()


def test_save_campaign_comparison_plots_noop_on_empty_input(tmp_path):
    save_campaign_comparison_plots({}, tmp_path)
    assert not (tmp_path / "campaign_convergence_curves.png").exists()
