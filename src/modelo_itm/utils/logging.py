import logging

from tqdm import tqdm

_CONFIGURED = False


class _TqdmLoggingHandler(logging.Handler):
    """Redirige los logs a traves de tqdm.write() para no romper las barras de
    progreso activas (M5: reemplaza los print() sueltos del training loop)."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def configure_logging(level: int = logging.INFO) -> None:
    """Configura el logger raiz una sola vez (idempotente)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = _TqdmLoggingHandler()
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
