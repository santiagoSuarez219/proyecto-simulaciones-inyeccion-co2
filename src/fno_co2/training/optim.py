import torch
import torch.nn as nn

from fno_co2.config import Config


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    """Crea el scheduler de LR segun cfg.lr_scheduler. None (default previo a
    M1: LR constante) desactiva el scheduler y esta funcion devuelve None."""
    if cfg.lr_scheduler is None:
        return None
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min
        )
    raise ValueError(
        f"lr_scheduler desconocido: {cfg.lr_scheduler!r}. Valores soportados: 'cosine', None."
    )


def build_param_groups(model: nn.Module, weight_decay: float):
    """Separa los parametros del modelo en dos grupos para AdamW: 'decay' (pesos
    de Conv2d/Linear/parametro espectral) y 'no_decay' (bias, embeddings —
    t_embed — y las capas gamma/beta de FiLMSpectralBlock, que modulan features
    como escala/desplazamiento). Practica estandar: no penalizar bias ni
    parametros de escala/desplazamiento con weight decay (M7)."""
    decay_params = []
    no_decay_params = []

    embedding_module_names = {
        name for name, module in model.named_modules() if isinstance(module, nn.Embedding)
    }

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module_path = param_name.rsplit(".", 1)[0] if "." in param_name else ""
        is_bias = param_name.endswith(".bias")
        is_embedding = module_path in embedding_module_names
        is_film_gate = module_path.endswith(".gamma") or module_path.endswith(".beta")
        if is_bias or is_embedding or is_film_gate:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
