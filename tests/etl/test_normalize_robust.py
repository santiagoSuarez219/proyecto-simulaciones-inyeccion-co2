"""B3 — clip_percentiles: outlier clipping opcional, no integrado
automaticamente en el pipeline (ver docstring en normalize.py)."""
import numpy as np

from modelo_itm.etl.normalize import clip_percentiles


def test_clip_percentiles_clips_outliers():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0], dtype=np.float64)
    clipped = clip_percentiles(values, p_low=1.0, p_high=90.0)
    assert clipped.max() < 1000.0
    assert clipped.min() >= values.min()


def test_clip_percentiles_no_outliers_stays_close_to_original():
    values = np.linspace(0.0, 1.0, 100)
    clipped = clip_percentiles(values, p_low=1.0, p_high=99.0)
    assert np.isclose(clipped.min(), np.percentile(values, 1.0))
    assert np.isclose(clipped.max(), np.percentile(values, 99.0))


def test_clip_percentiles_empty_array():
    values = np.array([], dtype=np.float64)
    result = clip_percentiles(values)
    assert result.size == 0


def test_clip_percentiles_ignores_nan_when_computing_bounds():
    values = np.array([1.0, 2.0, 3.0, np.nan, 1000.0], dtype=np.float64)
    clipped = clip_percentiles(values, p_low=0.0, p_high=75.0)
    # el bound superior debe calcularse solo sobre los valores finitos
    finite = values[np.isfinite(values)]
    expected_hi = np.percentile(finite, 75.0)
    assert np.isclose(np.nanmax(clipped[np.isfinite(clipped)]), expected_hi) or clipped[np.isfinite(clipped)].max() <= expected_hi + 1e-9


def test_clip_percentiles_all_nan_returns_unchanged():
    values = np.array([np.nan, np.nan], dtype=np.float64)
    result = clip_percentiles(values)
    assert np.isnan(result).all()
