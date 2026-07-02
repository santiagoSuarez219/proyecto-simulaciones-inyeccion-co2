import torch

from modelo_itm.config import Config
from modelo_itm.models.fno import PhysicalFNOArchitecture
from modelo_itm.training.checkpoint import build_run_signature, save_training_checkpoint, try_resume_training
from modelo_itm.training.optim import build_param_groups, build_scheduler


def _build_model_and_optimizer(cfg):
    model = PhysicalFNOArchitecture(time_steps=cfg.time_steps, h_dim=16, modes=4, dropout_p=cfg.dropout_p)
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg.weight_decay), lr=cfg.lr)
    return model, optimizer


def test_build_scheduler_none_when_disabled():
    cfg = Config(lr_scheduler=None)
    model, optimizer = _build_model_and_optimizer(cfg)
    assert build_scheduler(optimizer, cfg) is None


def test_build_scheduler_cosine_decays_lr_over_epochs():
    """M1: el LR debe variar a lo largo de las epocas con el scheduler activo
    (comportamiento previo: LR constante durante todo el entrenamiento)."""
    cfg = Config(lr_scheduler="cosine", lr=1e-3, lr_min=1e-6, epochs=10)
    model, optimizer = _build_model_and_optimizer(cfg)
    scheduler = build_scheduler(optimizer, cfg)
    assert scheduler is not None

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(cfg.epochs):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == 1e-3
    assert lrs[-1] == 1e-6  # CosineAnnealingLR con T_max=epochs llega a eta_min al final
    assert lrs != [lrs[0]] * len(lrs), "el LR no vario — sigue siendo constante"
    # Monotonamente no creciente (forma de coseno decayendo)
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1))


def test_build_scheduler_unknown_raises():
    cfg = Config(lr_scheduler="not_a_real_scheduler")
    model, optimizer = _build_model_and_optimizer(cfg)
    try:
        build_scheduler(optimizer, cfg)
        assert False, "deberia haber lanzado ValueError"
    except ValueError:
        pass


def test_checkpoint_roundtrip_preserves_scheduler_state(tmp_path):
    """M1: guardar/reanudar un checkpoint debe restaurar el punto exacto del
    ciclo del scheduler, no reiniciarlo desde el LR inicial."""
    cfg = Config(lr_scheduler="cosine", lr=1e-3, lr_min=1e-6, epochs=10, time_steps=4)
    train_path = tmp_path / "train"
    val_path = tmp_path / "val"
    train_path.mkdir()
    val_path.mkdir()
    run_signature = build_run_signature(cfg, train_path, val_path)

    # --- Sesion 1: entrena 3 "epocas" (solo avanza el scheduler) y guarda ---
    model_a, optimizer_a = _build_model_and_optimizer(cfg)
    scheduler_a = build_scheduler(optimizer_a, cfg)
    for _ in range(3):
        scheduler_a.step()
    lr_after_3_epochs = optimizer_a.param_groups[0]["lr"]

    ckpt_path = tmp_path / "latest.pt"
    save_training_checkpoint(
        ckpt_path, model_a, optimizer_a, cfg, epoch=3, best_val_loss=0.5,
        metrics_row={"val_loss": 0.5}, run_signature=run_signature, scheduler=scheduler_a,
    )

    # --- Sesion 2: modelo/optimizer/scheduler nuevos, reanuda desde el checkpoint ---
    model_b, optimizer_b = _build_model_and_optimizer(cfg)
    scheduler_b = build_scheduler(optimizer_b, cfg)
    start_epoch, best_val_loss, _, resumed, reasons, _ = try_resume_training(
        ckpt_path, model_b, optimizer_b, torch.device("cpu"), run_signature, scheduler=scheduler_b,
    )

    assert resumed, f"resume fallo: {reasons}"
    assert start_epoch == 4
    # El LR restaurado en el optimizer (via optimizer_state_dict) debe coincidir
    # exactamente con el que tenia la sesion 1 tras 3 epocas.
    assert optimizer_b.param_groups[0]["lr"] == lr_after_3_epochs

    # El scheduler restaurado debe seguir el mismo camino que el original si
    # avanza una epoca mas en ambas sesiones.
    scheduler_a.step()
    scheduler_b.step()
    assert optimizer_a.param_groups[0]["lr"] == optimizer_b.param_groups[0]["lr"]


def test_checkpoint_resume_without_scheduler_state_does_not_abort(tmp_path):
    """Un checkpoint guardado sin scheduler (scheduler=None) debe poder
    reanudarse igual, aunque la sesion actual si tenga scheduler activo."""
    cfg = Config(lr_scheduler="cosine", lr=1e-3, epochs=10, time_steps=4)
    train_path = tmp_path / "train"
    val_path = tmp_path / "val"
    train_path.mkdir()
    val_path.mkdir()
    run_signature = build_run_signature(cfg, train_path, val_path)

    model_a, optimizer_a = _build_model_and_optimizer(cfg)
    ckpt_path = tmp_path / "latest.pt"
    save_training_checkpoint(
        ckpt_path, model_a, optimizer_a, cfg, epoch=1, best_val_loss=0.5,
        metrics_row={"val_loss": 0.5}, run_signature=run_signature, scheduler=None,
    )

    model_b, optimizer_b = _build_model_and_optimizer(cfg)
    scheduler_b = build_scheduler(optimizer_b, cfg)
    start_epoch, _, _, resumed, reasons, _ = try_resume_training(
        ckpt_path, model_b, optimizer_b, torch.device("cpu"), run_signature, scheduler=scheduler_b,
    )
    assert resumed, f"resume fallo: {reasons}"
    assert start_epoch == 2
