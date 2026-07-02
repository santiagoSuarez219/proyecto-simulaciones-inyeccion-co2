import random

import numpy as np
import torch
import torch.nn as nn

from modelo_itm.config import DEFAULT_DEVICE


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    requested = (requested or DEFAULT_DEVICE).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "gpu":
        requested = "cuda"

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Se pidio usar CUDA/GPU, pero esta instalacion de PyTorch no detecta CUDA. "
                "Revisa que estes ejecutando este archivo con el mismo Python que tiene PyTorch CUDA "
                "o instala una build de PyTorch con soporte CUDA."
            )

        cuda_index = 0 if device.index is None else device.index
        if cuda_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Se pidio {device}, pero solo hay {torch.cuda.device_count()} GPU(s) CUDA disponibles."
            )

        torch.cuda.set_device(cuda_index)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device(f"cuda:{cuda_index}")

    if device.type != "cpu":
        raise ValueError(f"Dispositivo no soportado: {requested}. Usa cuda, cuda:0, gpu, auto o cpu.")
    return device


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = 0 if device.index is None else device.index
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024**3)
        return f"cuda:{idx} | {props.name} | {total_gb:.1f} GB"
    return str(device)


def assert_model_on_device(model: nn.Module, device: torch.device):
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        return

    if model_device.type != device.type:
        raise RuntimeError(f"El modelo quedo en {model_device}, pero se esperaba {device}.")
    if device.type == "cuda" and model_device.index != device.index:
        raise RuntimeError(f"El modelo quedo en {model_device}, pero se esperaba {device}.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
