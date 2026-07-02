import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score

from modelo_itm.training.metrics import (
    finalize_global_regression_metrics,
    init_global_regression_accumulators,
    update_global_regression_accumulators,
)


def _reference_r2_rmse(pred: np.ndarray, gt: np.ndarray):
    r2 = float(r2_score(gt.reshape(-1), pred.reshape(-1)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return r2, rmse


def test_global_accumulation_matches_sklearn_single_batch():
    torch.manual_seed(0)
    pred = torch.randn(4, 61, 10, 10)
    gt = torch.randn(4, 61, 10, 10)

    acc = init_global_regression_accumulators(("sf",))
    update_global_regression_accumulators(acc, "sf", pred, gt)
    result = finalize_global_regression_metrics(acc)

    ref_r2, ref_rmse = _reference_r2_rmse(pred.numpy(), gt.numpy())
    assert result["sf_r2"] == pytest.approx(ref_r2, abs=1e-5)
    assert result["sf_rmse"] == pytest.approx(ref_rmse, abs=1e-5)


def test_global_accumulation_matches_reference_with_uneven_batches():
    """Simula evaluate_epoch: N batches de tamaño desigual (el último más chico),
    acumulados incrementalmente. Debe coincidir con el calculo sobre el dataset
    completo de una sola vez — esto es exactamente lo que C3 corrige."""
    torch.manual_seed(42)
    batch_sizes = [4, 4, 4, 1]  # último batch parcial, como ocurre en la práctica
    preds = [torch.randn(b, 61, 8, 8) for b in batch_sizes]
    gts = [torch.randn(b, 61, 8, 8) for b in batch_sizes]

    acc = init_global_regression_accumulators(("vd",))
    for pred, gt in zip(preds, gts):
        update_global_regression_accumulators(acc, "vd", pred, gt)
    result = finalize_global_regression_metrics(acc)

    full_pred = torch.cat(preds, dim=0).numpy()
    full_gt = torch.cat(gts, dim=0).numpy()
    ref_r2, ref_rmse = _reference_r2_rmse(full_pred, full_gt)

    assert result["vd_r2"] == pytest.approx(ref_r2, abs=1e-5)
    assert result["vd_rmse"] == pytest.approx(ref_rmse, abs=1e-5)


def test_global_accumulation_differs_from_naive_batch_averaging():
    """Confirma que el bug de C3 (promediar R2/RMSE por batch) da un resultado
    distinto del correcto cuando los batches tienen tamaños desiguales."""
    torch.manual_seed(7)
    batch_sizes = [8, 8, 8, 2]
    preds = [torch.randn(b, 61, 6, 6) for b in batch_sizes]
    gts = [torch.randn(b, 61, 6, 6) for b in batch_sizes]

    acc = init_global_regression_accumulators(("sf",))
    naive_r2_values = []
    naive_rmse_values = []
    for pred, gt in zip(preds, gts):
        update_global_regression_accumulators(acc, "sf", pred, gt)
        batch_r2 = r2_score(gt.numpy().reshape(-1), pred.numpy().reshape(-1))
        batch_rmse = np.sqrt(np.mean((pred.numpy() - gt.numpy()) ** 2))
        naive_r2_values.append(batch_r2)
        naive_rmse_values.append(batch_rmse)

    correct = finalize_global_regression_metrics(acc)
    naive_r2 = float(np.mean(naive_r2_values))
    naive_rmse = float(np.mean(naive_rmse_values))

    assert correct["sf_r2"] != pytest.approx(naive_r2, abs=1e-6)
    assert correct["sf_rmse"] != pytest.approx(naive_rmse, abs=1e-6)


def test_finalize_global_regression_metrics_empty_accumulator():
    acc = init_global_regression_accumulators(("sf", "vd"))
    result = finalize_global_regression_metrics(acc)
    assert result["sf_r2"] == 0.0
    assert result["sf_rmse"] == 0.0
    assert result["vd_r2"] == 0.0
    assert result["vd_rmse"] == 0.0
