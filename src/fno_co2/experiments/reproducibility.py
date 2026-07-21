"""Captura de reproducibilidad de una campaña (spec-004 Fase 2).

Al iniciar una campaña, `capture_reproducibility` escribe (una sola vez, atómicamente)
todo lo necesario para reconstruir meses después el entorno, el código exacto y los datos
exactos que produjeron cada número: git hash + dirty, `pip freeze` + versiones, checksum
del split, copia de cada config por-variante, y un `campaign_manifest.json` que enlaza
todo lo anterior (spec-004 §1.4).
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from fno_co2.experiments.campaign_config import CampaignConfig, compute_file_checksum


# ==========================================
# Escritura atómica (write-to-temp + rename)
# ==========================================
def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_json(path: Path, data) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


# ==========================================
# Captura individual
# ==========================================
def capture_git_info(repo_root: Path) -> dict:
    def _run(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()

    commit_hash = _run(["rev-parse", "HEAD"])
    dirty_status = _run(["status", "--porcelain"])
    return {"commit_hash": commit_hash, "is_dirty": bool(dirty_status)}


def capture_environment_info() -> str:
    lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"torch.cuda.is_available(): {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda: {torch.version.cuda}")
        lines.append(f"cudnn: {torch.backends.cudnn.version()}")
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")

    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True,
    ).stdout

    lines.append("")
    lines.append("--- pip freeze ---")
    lines.append(pip_freeze.strip())
    return "\n".join(lines)


def copy_config_snapshots(campaign: CampaignConfig, dest_dir: Path) -> dict[str, str]:
    """Copia (no referencia) el config de cada variante — inmune a ediciones posteriores
    del YAML original (spec-004 §1.4)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths: dict[str, str] = {}
    for variant in campaign.variants:
        dest = dest_dir / f"{variant.name}.yaml"
        shutil.copy2(variant.config_path, dest)
        snapshot_paths[variant.name] = str(dest)
    return snapshot_paths


def build_campaign_manifest(
    campaign: CampaignConfig,
    git_info: dict,
    split_checksum: str,
    config_snapshot_paths: dict[str, str],
    reproducibility_dir: Path,
    timestamp: str,
) -> dict:
    return {
        "campaign_name": campaign.campaign_name,
        "description": campaign.description,
        "seeds": campaign.seeds,
        "variants": [
            {
                "name": variant.name,
                "config_path": str(variant.config_path),
                "success_criterion": variant.success_criterion,
            }
            for variant in campaign.variants
        ],
        "tracking_backend": campaign.tracking_backend,
        "git": git_info,
        "split_checksum": split_checksum,
        "config_snapshots": config_snapshot_paths,
        "started_at": timestamp,
        "reproducibility_dir": str(reproducibility_dir),
    }


# ==========================================
# Orquestador (spec-004 §1.4, llamar una vez al iniciar la campaña)
# ==========================================
def capture_reproducibility(
    campaign: CampaignConfig,
    *,
    outputs_root: Path = Path("outputs/campaigns"),
    split_path: Path = Path("reports/train_test_split_80_20.csv"),
    repo_root: Path = Path("."),
    timestamp: str | None = None,
) -> Path:
    """Escribe `outputs/campaigns/<name>/reproducibility/{git.json,environment.txt,
    split.sha256,configs/}` + `campaign_manifest.json`, todo atómicamente. Retorna la ruta
    de `reproducibility/`. El checksum de `split.sha256` es el que la guarda de
    comparabilidad del preflight (`run_preflight`) fija y compara en corridas futuras."""
    campaign_dir = outputs_root / campaign.campaign_name
    reproducibility_dir = campaign_dir / "reproducibility"

    git_info = capture_git_info(repo_root)
    atomic_write_json(reproducibility_dir / "git.json", git_info)

    atomic_write_text(reproducibility_dir / "environment.txt", capture_environment_info())

    split_checksum = compute_file_checksum(split_path)
    atomic_write_text(reproducibility_dir / "split.sha256", split_checksum)

    config_snapshot_paths = copy_config_snapshots(campaign, reproducibility_dir / "configs")

    manifest = build_campaign_manifest(
        campaign,
        git_info,
        split_checksum,
        config_snapshot_paths,
        reproducibility_dir,
        timestamp or datetime.now().isoformat(),
    )
    atomic_write_json(campaign_dir / "campaign_manifest.json", manifest)

    return reproducibility_dir
