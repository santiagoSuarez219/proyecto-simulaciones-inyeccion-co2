import logging

from modelo_itm.utils.logging import _TqdmLoggingHandler, configure_logging, get_logger


def test_get_logger_returns_named_logger():
    logger = get_logger("modelo_itm.test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "modelo_itm.test_module"


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
    logger = get_logger("modelo_itm.test_emit")
    logger.info("mensaje de prueba %s", 42)  # no debe lanzar excepciones
