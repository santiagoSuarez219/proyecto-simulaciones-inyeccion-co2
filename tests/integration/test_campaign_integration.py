"""Test de integración corto de campaña (spec-004 Fase 6).

Corre la CLI real (`scripts/run_campaign.py` + `scripts/aggregate_campaign.py`, sin mocks)
como subprocesos, contra los datos reales del repo: una campaña mínima con solo `baseline`,
3 seeds (mínimo que exige el preflight, spec-001 Fase 6 — el borrador original de este spec
pedía 2, inconsistente con esa guarda ya implementada), 1 época, overfit de 1 muestra
(`--overfit-sample-idx 0`, `--device cpu` — esta workstation no tiene CUDA). Verifica que
produzca `campaign_state.json` consistente, `run.done` por corrida, manifiesto de
reproducibilidad y un `campaign_report.md` — el camino de código completo, no una simulación.

**Restricción arquitectónica descubierta al escribir este test (documentada, no un bug
introducido aquí):** `train.py::resolve_config` deriva su `output_dir` real con el literal
relativo `"outputs/<experiment_name>/seed_<seed>"` (relativo al `cwd` del proceso), sin
enterarse del `outputs_root` que `run_experiment.py`/`run_campaign()` usan para su propia
contabilidad (`run_manifest.json`, `campaign_state.json`, `run.done`). Como `train.py` no se
puede modificar (spec-004 criterios de aceptación), una campaña real **siempre** escribe los
artefactos de entrenamiento bajo `<cwd>/outputs/campaigns/<name>/...` — el flag
`--outputs-root` solo redirige la contabilidad propia de la campaña, no las escrituras reales
de `train.py`. Este test corre con `cwd` = raíz del repo (para que preflight resuelva
`data/processed`/el split reales) y limpia expresamente su directorio de salida al terminar.

Marcado `@pytest.mark.slow`: excluido de `pytest tests/ -m "not slow"`, corre bajo demanda.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_NAME = "spec004_fase6_integration_smoke"


@pytest.mark.slow
def test_minimal_campaign_end_to_end(tmp_path):
    campaign_dir = REPO_ROOT / "outputs" / "campaigns" / CAMPAIGN_NAME
    assert not campaign_dir.exists(), (
        f"{campaign_dir} ya existe de una corrida anterior sin limpiar — bórrala manualmente "
        "antes de re-correr este test (por seguridad, el test no sobrescribe sin --resume)."
    )

    campaign_yaml = tmp_path / "mini_campaign.yaml"
    campaign_yaml.write_text(
        yaml.safe_dump({
            "campaign_name": CAMPAIGN_NAME,
            "description": "Campaña mínima de integración (spec-004 Fase 6).",
            "seeds": [42, 43, 44],
            "variants": [
                {
                    "name": "baseline",
                    "config": str(REPO_ROOT / "configs" / "experiments" / "baseline.yaml"),
                    "success_criterion": "referencia (línea base); no se evalúa contra sí misma",
                },
            ],
            "tracking": {"backend": "file"},
        }),
        encoding="utf-8",
    )
    docs_path = tmp_path / "docs_test.md"

    try:
        run_result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "run_campaign.py"),
                "--config", str(campaign_yaml),
                "--yes",
                "--extra-args", "--epochs 1 --overfit-sample-idx 0 --device cpu",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        assert run_result.returncode == 0, run_result.stdout + run_result.stderr

        state = json.loads((campaign_dir / "campaign_state.json").read_text(encoding="utf-8"))
        for seed in (42, 43, 44):
            assert state["jobs"][f"baseline/seed_{seed}"]["status"] == "completed"
            assert (campaign_dir / "baseline" / f"seed_{seed}" / "run.done").exists()

        manifest = json.loads((campaign_dir / "campaign_manifest.json").read_text(encoding="utf-8"))
        assert manifest["campaign_name"] == CAMPAIGN_NAME
        assert (campaign_dir / "reproducibility" / "split.sha256").exists()
        assert (campaign_dir / "reproducibility" / "configs" / "baseline.yaml").exists()

        aggregate_result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "aggregate_campaign.py"),
                "--config", str(campaign_yaml),
                "--docs-path", str(docs_path),
            ],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        assert aggregate_result.returncode == 0, aggregate_result.stdout + aggregate_result.stderr

        assert (campaign_dir / "campaign_report.md").exists()
        assert "<!-- experiment: baseline -->" in docs_path.read_text(encoding="utf-8")

        # --resume debe converger al mismo estado sin re-ejecutar nada (idempotencia, spec-004 §2)
        resume_result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "run_campaign.py"),
                "--config", str(campaign_yaml),
                "--yes", "--resume",
                "--extra-args", "--epochs 1 --overfit-sample-idx 0 --device cpu",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        assert resume_result.returncode == 0, resume_result.stdout + resume_result.stderr
        state_after_resume = json.loads((campaign_dir / "campaign_state.json").read_text(encoding="utf-8"))
        for seed in (42, 43, 44):
            assert state_after_resume["jobs"][f"baseline/seed_{seed}"].get("skipped") is True
    finally:
        shutil.rmtree(campaign_dir, ignore_errors=True)
