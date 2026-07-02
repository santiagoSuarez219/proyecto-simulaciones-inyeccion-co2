from pathlib import Path

import torch
from torch.utils.data import Dataset

from modelo_itm.config import _LAYER_RE


def _k(p: Path):
    m = _LAYER_RE.search(p.name)
    return int(m.group(1)) if m else None


def load_pt(p: Path):
    x = torch.load(p, map_location="cpu")
    if isinstance(x, dict):
        for v in x.values():
            if torch.is_tensor(v):
                return v.float()
        raise ValueError(f"Archivo dict sin tensores: {p}")
    if not torch.is_tensor(x):
        raise ValueError(f"Archivo no tensor: {p}")
    return x.float()


def load_injection_series(inj_paths, time_steps: int):
    """Carga series de inyeccion ya normalizadas por el ETL (min-max con
    global_stats de train — mismo mecanismo que la normalizacion de C1),
    solo alinea longitud a time_steps. No se re-normaliza aqui (ver A2:
    antes se aplicaba log1p + reescalado local por muestra encima de
    valores ya normalizados en [0,1] por el ETL, produciendo una escala
    efectiva dependiente de cada muestra individual)."""
    if not inj_paths:
        return torch.zeros(time_steps, 2, dtype=torch.float32)

    series = []
    for p in sorted(inj_paths)[:2]:
        t = load_pt(p).float().reshape(-1)
        if t.numel() < time_steps:
            pad = torch.zeros(time_steps - t.numel(), dtype=torch.float32)
            t = torch.cat([t, pad], dim=0)
        else:
            t = t[:time_steps]
        series.append(t)

    while len(series) < 2:
        series.append(torch.zeros(time_steps, dtype=torch.float32))

    inj = torch.stack(series[:2], dim=1)
    # Red de seguridad: el ETL deberia producir valores limpios, pero se
    # sanitiza por si llegan NaN/Inf de datos de entrada corruptos.
    inj = torch.nan_to_num(inj, nan=0.0, posinf=0.0, neginf=0.0)
    return inj


class DatasetLayers(Dataset):
    def __init__(self, root, max_layer=60):
        self.samples = []
        self.max_layer = int(max_layer)
        self._inj_cache = {}
        root = Path(root)
        if not root.exists():
            return

        for case in root.iterdir():
            if not case.is_dir():
                continue

            static = {
                "AFI": case / "afi_layer_cubes",
                "COH": case / "cohesion_layer_cubes",
                "PERM": case / "permeability_layer_cubes",
                "PORO": case / "porosity_layer_cubes",
            }
            target = case / "layer_cubes"
            inj_dir = case / "injection_name_tensors"

            if not target.exists() or not all(v.exists() for v in static.values()):
                continue

            idx_static = {k: {_k(f): f for f in v.glob("*.pt")} for k, v in static.items()}
            idx_target = {_k(f): f for f in target.glob("*.pt")}
            inj_paths = sorted(inj_dir.glob("*.pt")) if inj_dir.exists() else []

            for k in range(1, 98):
                if k in idx_target and all(k in idx_static[p] for p in static):
                    self.samples.append({
                        "case": case.name,
                        "k": k,
                        "static": {p: idx_static[p][k] for p in static},
                        "target": idx_target[k],
                        "inj_paths": inj_paths,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = torch.stack(
            [load_pt(s["static"][k]).squeeze() for k in ["AFI", "COH", "PERM", "PORO"]],
            dim=0,
        ).float()
        y = load_pt(s["target"]).permute(1, 0, 2, 3)[: self.max_layer + 1].float()
        depth = torch.tensor([(s["k"] - 1) / 96.0], dtype=torch.float32)
        inj_key = tuple(s.get("inj_paths", []))
        inj = self._inj_cache.get(inj_key)
        if inj is None or inj.size(0) != y.shape[0]:
            inj = load_injection_series(s.get("inj_paths", []), y.shape[0])
            self._inj_cache[inj_key] = inj
        return x, depth, inj, y
