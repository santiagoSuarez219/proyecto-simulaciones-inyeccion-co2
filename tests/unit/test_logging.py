import logging

from fno_co2.utils.logging import EmitOnce, _TqdmLoggingHandler, configure_logging, get_logger


def test_get_logger_returns_named_logger():
    logger = get_logger("fno_co2.test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "fno_co2.test_module"


def test_configure_logging_is_idempotent():
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    configure_logging()
    configure_logging()
    configure_logging()

    tqdm_handlers = [h for h in root.handlers if isinstance(h, _TqdmLoggingHandler)]
    assert len(tqdm_handlers) == 1, "configure_logging no debe duplicar handlers al llamarse varias veces"


def test_configure_logging_installs_tqdm_handler():
    configure_logging()
    root = logging.getLogger()
    assert any(isinstance(h, _TqdmLoggingHandler) for h in root.handlers)


def test_logger_emits_without_raising(capsys):
    logger = get_logger("fno_co2.test_emit")
    logger.info("mensaje de prueba %s", 42)  # no debe lanzar excepciones


def test_emit_once_only_true_first_time():
    """B1: reemplaza las variables globales _XXX_EMITTED sueltas."""
    once = EmitOnce()
    assert once.should_emit("warning_a") is True
    assert once.should_emit("warning_a") is False
    assert once.should_emit("warning_a") is False


def test_emit_once_tracks_keys_independently():
    once = EmitOnce()
    assert once.should_emit("a") is True
    assert once.should_emit("b") is True
    assert once.should_emit("a") is False
    assert once.should_emit("b") is False


def test_emit_once_reset_specific_key():
    once = EmitOnce()
    once.should_emit("a")
    once.should_emit("b")
    once.reset("a")
    assert once.should_emit("a") is True
    assert once.should_emit("b") is False


def test_emit_once_reset_all():
    once = EmitOnce()
    once.should_emit("a")
    once.should_emit("b")
    once.reset()
    assert once.should_emit("a") is True
    assert once.should_emit("b") is True


def test_emit_once_instances_are_independent():
    """Cada instancia de EmitOnce tiene su propio estado — a diferencia de una
    variable global de modulo compartida entre todas las llamadas."""
    once_a = EmitOnce()
    once_b = EmitOnce()
    once_a.should_emit("x")
    assert once_b.should_emit("x") is True
