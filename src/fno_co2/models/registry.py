import importlib

import torch.nn as nn

from fno_co2.config import Config
from fno_co2.models.fno import PhysicalFNOArchitecture

BASELINE_VARIANT = "fno_baseline"


def _build_baseline(cfg: Config) -> nn.Module:
    return PhysicalFNOArchitecture(
        time_steps=cfg.time_steps,
        in_c=5,
        h_dim=cfg.hidden_dim,
        modes=cfg.spectral_modes,
        cond_dim=128,
        dropout_p=cfg.dropout_p,
        use_group_norm=cfg.use_group_norm,
    )


def build_model(cfg: Config) -> nn.Module:
    """Despacha la arquitectura por `cfg.model_variant` (spec-001 Fase 3).

    `"fno_baseline"` construye la línea base (`PhysicalFNOArchitecture`) directamente. Para
    cualquier otro valor, busca el módulo `fno_co2.models.variants.<model_variant>` y llama a
    su función `build(cfg) -> nn.Module` — así una variante nueva vive en su propio archivo
    (ver `src/fno_co2/models/variants/`) sin tocar este registry ni la línea base.
    """
    variant = cfg.model_variant

    if variant == BASELINE_VARIANT:
        return _build_baseline(cfg)

    try:
        module = importlib.import_module(f"fno_co2.models.variants.{variant}")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"model_variant desconocida: '{variant}'. Esperaba '{BASELINE_VARIANT}' o un "
            f"módulo en fno_co2.models.variants.{variant} (ver spec-001 Fase 3)."
        ) from exc

    build_fn = getattr(module, "build", None)
    if build_fn is None:
        raise ValueError(
            f"fno_co2.models.variants.{variant} no define una función build(cfg) -> nn.Module."
        )
    return build_fn(cfg)
