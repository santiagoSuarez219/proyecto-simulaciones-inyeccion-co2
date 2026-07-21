import hashlib
import json
import subprocess

import pytest

from fno_co2.experiments.campaign_config import CampaignConfig, CampaignVariant
from fno_co2.experiments.reproducibility import (
    atomic_write_json,
    atomic_write_text,
    capture_environment_info,
    capture_git_info,
    capture_reproducibility,
    copy_config_snapshots,
)


def _run_git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def isolated_git_repo(tmp_path):
    """Repo git aislado y desechable en tmp_path — nunca toca el repo real del proyecto."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _run_git(["init"], repo_dir)
    _run_git(["config", "user.email", "test@example.com"], repo_dir)
    _run_git(["config", "user.name", "Test"], repo_dir)
    (repo_dir / "file.txt").write_text("contenido inicial")
    _run_git(["add", "file.txt"], repo_dir)
    _run_git(["commit", "-m", "commit inicial"], repo_dir)
    return repo_dir


def test_capture_git_info_reports_commit_hash_and_clean_tree(isolated_git_repo):
    info = capture_git_info(isolated_git_repo)
    expected_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=isolated_git_repo, capture_output=True, text=True, check=True,
    ).stdout.strip()

    assert info["commit_hash"] == expected_hash
    assert info["is_dirty"] is False


def test_capture_git_info_detects_dirty_tree(isolated_git_repo):
    (isolated_git_repo / "file.txt").write_text("modificado sin commitear")
    info = capture_git_info(isolated_git_repo)
    assert info["is_dirty"] is True


def test_capture_environment_info_includes_python_and_torch_versions():
    env_text = capture_environment_info()
    assert "python:" in env_text
    assert "torch:" in env_text
    assert "pip freeze" in env_text


def test_atomic_write_text_and_json_roundtrip(tmp_path):
    text_path = tmp_path / "out" / "split.sha256"
    atomic_write_text(text_path, "abc123")
    assert text_path.read_text(encoding="utf-8") == "abc123"
    assert not text_path.with_name(text_path.name + ".tmp").exists()

    json_path = tmp_path / "out" / "git.json"
    atomic_write_json(json_path, {"commit_hash": "abc", "is_dirty": False})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"commit_hash": "abc", "is_dirty": False}


def _make_campaign(tmp_path):
    config_a = tmp_path / "configs" / "baseline.yaml"
    config_a.parent.mkdir(parents=True, exist_ok=True)
    config_a.write_text("model_variant: fno_baseline\n", encoding="utf-8")

    config_b = tmp_path / "configs" / "unet_film.yaml"
    config_b.write_text("model_variant: unet_film\n", encoding="utf-8")

    return CampaignConfig(
        campaign_name="test_campaign",
        description="campaña de prueba",
        seeds=[42, 43, 44],
        variants=[
            CampaignVariant(name="baseline", config_path=config_a, success_criterion="referencia"),
            CampaignVariant(
                name="unet_film", config_path=config_b,
                success_criterion={"metric": "val_sf_r2", "op": ">=", "threshold": 0.9},
            ),
        ],
    )


def test_copy_config_snapshots_copies_each_variant_config(tmp_path):
    campaign = _make_campaign(tmp_path)
    dest_dir = tmp_path / "snapshots"

    snapshot_paths = copy_config_snapshots(campaign, dest_dir)

    assert set(snapshot_paths) == {"baseline", "unet_film"}
    for variant_name, path_str in snapshot_paths.items():
        assert (dest_dir / f"{variant_name}.yaml").exists()
        assert (dest_dir / f"{variant_name}.yaml").read_text(encoding="utf-8") == (
            campaign.variants[0].config_path.read_text(encoding="utf-8")
            if variant_name == "baseline" else campaign.variants[1].config_path.read_text(encoding="utf-8")
        )


def test_capture_reproducibility_writes_full_manifest_and_artifacts(tmp_path, isolated_git_repo):
    campaign = _make_campaign(tmp_path)

    split_path = tmp_path / "split.csv"
    split_path.write_text("sim_id,split\n001,train\n", encoding="utf-8")
    expected_checksum = hashlib.sha256(split_path.read_bytes()).hexdigest()

    outputs_root = tmp_path / "outputs" / "campaigns"

    reproducibility_dir = capture_reproducibility(
        campaign,
        outputs_root=outputs_root,
        split_path=split_path,
        repo_root=isolated_git_repo,
        timestamp="2026-07-16T00:00:00",
    )

    assert reproducibility_dir == outputs_root / "test_campaign" / "reproducibility"
    assert (reproducibility_dir / "git.json").exists()
    assert (reproducibility_dir / "environment.txt").exists()
    assert (reproducibility_dir / "split.sha256").read_text(encoding="utf-8") == expected_checksum
    assert (reproducibility_dir / "configs" / "baseline.yaml").exists()
    assert (reproducibility_dir / "configs" / "unet_film.yaml").exists()

    manifest_path = outputs_root / "test_campaign" / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["campaign_name"] == "test_campaign"
    assert manifest["seeds"] == [42, 43, 44]
    assert manifest["split_checksum"] == expected_checksum
    assert manifest["git"]["is_dirty"] is False
    assert set(manifest["config_snapshots"]) == {"baseline", "unet_film"}
    assert manifest["started_at"] == "2026-07-16T00:00:00"


def test_capture_reproducibility_marks_dirty_tree_from_manifest(tmp_path, isolated_git_repo):
    (isolated_git_repo / "file.txt").write_text("cambio sin commitear")
    campaign = _make_campaign(tmp_path)
    split_path = tmp_path / "split.csv"
    split_path.write_text("sim_id,split\n001,train\n", encoding="utf-8")

    capture_reproducibility(
        campaign,
        outputs_root=tmp_path / "outputs" / "campaigns",
        split_path=split_path,
        repo_root=isolated_git_repo,
    )

    manifest_path = tmp_path / "outputs" / "campaigns" / "test_campaign" / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["git"]["is_dirty"] is True
