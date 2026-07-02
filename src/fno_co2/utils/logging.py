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


class EmitOnce:
    """Encapsula el patron 'emitir un mensaje una sola vez', reemplazando
    variables globales sueltas de tipo `_XXX_EMITTED` con `global` (B1:
    `_MC_DROPOUT_WARNING_EMITTED` en inference/uncertainty.py,
    `_CUDA_BATCH_REPORT_EMITTED` en training/loop.py)."""

    def __init__(self):
        self._emitted: set[str] = set()

    def should_emit(self, key: str) -> bool:
        """True la primera vez que se llama con esta key; False despues."""
        if key in self._emitted:
            return False
        self._emitted.add(key)
        return True

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._emitted.clear()
        else:
            self._emitted.discard(key)
