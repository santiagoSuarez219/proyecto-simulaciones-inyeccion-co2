import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from modelo_itm.config import Config
from modelo_itm.data.dataset import DatasetLayers


def resolve_num_workers(requested: int | None) -> int:
    if requested is not None:
        return max(0, int(requested))

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return min(8, max(2, cpu_count // 2))


def build_loader(dataset, cfg: Config, device: torch.device, shuffle: bool):
    num_workers = resolve_num_workers(cfg.num_workers)
    kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
    return DataLoader(dataset, **kwargs), num_workers


def resolve_dir(path_value: str | None, data_root: str, fallbacks, label: str) -> Path:
    root = Path(data_root)
    candidates = []

    if path_value:
        p = Path(path_value)
        if not p.is_absolute():
            p = root / p
        candidates.append(p)
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"No existe el directorio de {label}: {p}")

    for name in fallbacks:
        p = root / name
        candidates.append(p)
        if p.exists() and p.is_dir():
            return p

    searched = ", ".join(str(x) for x in candidates) or "(sin candidatos)"
    raise FileNotFoundError(f"No pude resolver el directorio de {label}. Busque en: {searched}")


def build_datasets(cfg: Config):
    train_path = resolve_dir(cfg.train_dir, cfg.data_root, ("train",), "train")
    train_ds_full = DatasetLayers(train_path, max_layer=cfg.time_steps - 1)
    if len(train_ds_full) == 0:
        raise FileNotFoundError(f"No se encontraron datos en {train_path}")

    if cfg.overfit_sample_idx is not None:
        idx = int(cfg.overfit_sample_idx)
        if idx < 0 or idx >= len(train_ds_full):
            raise IndexError(
                f"overfit_sample_idx={idx} fuera de rango. Dataset size={len(train_ds_full)}"
            )
        train_ds = Subset(train_ds_full, [idx])
        val_ds = Subset(train_ds_full, [idx])
        return train_ds, val_ds, train_path, train_path

    if cfg.val_dir is not None:
        val_path = resolve_dir(cfg.val_dir, cfg.data_root, (), "validacion")
        val_ds = DatasetLayers(val_path, max_layer=cfg.time_steps - 1)
        return train_ds_full, val_ds, train_path, val_path

    default_test = Path(cfg.data_root) / "test"
    if default_test.exists() and default_test.is_dir():
        val_ds = DatasetLayers(default_test, max_layer=cfg.time_steps - 1)
        return train_ds_full, val_ds, train_path, default_test

    n_train = max(1, int(0.9 * len(train_ds_full)))
    n_val = len(train_ds_full) - n_train
    if n_val == 0:
        n_train = len(train_ds_full) - 1
        n_val = 1
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(train_ds_full, [n_train, n_val], generator=gen)
    return train_ds, val_ds, train_path, train_path
