from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_unified_gpu_bootstrap_dry_run_is_non_mutating(tmp_path: Path) -> None:
    environment = tmp_path / ".venv"
    dependencies = tmp_path / ".deps"
    env = os.environ.copy()
    env.update(
        {
            "CDLAM_DEPS_DIR": str(dependencies),
            "CDLAM_VENV": str(environment),
            "PYTHON": sys.executable,
        }
    )
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "setup.sh"),
            "--accept-base-license",
            "--dry-run",
            "--gpu",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if sys.version_info[:2] != (3, 10):
        assert result.returncode == 2
        assert "requires CPython 3.10" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    assert "create unified GPU environment" in result.stdout
    assert "torch 2.7.0+cu128, CUDA 12.8" in result.stdout
    assert "CUDA optimizer smokes" in result.stdout
    assert not environment.exists()
    assert not dependencies.exists()


def test_setup_requires_explicit_license_acceptance() -> None:
    env = os.environ.copy()
    env.pop("CDLAM_ACCEPT_BASE_LICENSE", None)
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "setup.sh"), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "--accept-base-license" in result.stderr


def test_root_run_entrypoint_exposes_help_without_environment() -> None:
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "run.sh"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "train-smoke" in result.stdout
    assert "pipeline" in result.stdout


def test_gpu_contract_is_explicit_in_code_and_config() -> None:
    smoke = (ROOT / "scripts/gpu_smoke.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "scripts/bootstrap.sh").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    runtime = (ROOT / "configs/model_runtime.lock.json").read_text(encoding="utf-8")
    profile = (ROOT / "configs/runtime.example.json").read_text(encoding="utf-8")

    assert 'EXPECTED_TORCH = "2.7.0+cu128"' in smoke
    assert 'EXPECTED_CUDA = "12.8"' in smoke
    assert "--editable" not in bootstrap
    assert "--device" not in bootstrap
    assert "--device" not in makefile
    assert 'CDLAM_TEST_ACWM_ROOT="$DEPS_ROOT/acwm-runtime"' in bootstrap
    assert '"nvidia_driver_minimum": "570.124.06"' in runtime
    assert '"python": ".venv/bin/python"' in profile
    assert '"torchrun": ".venv/bin/torchrun"' in profile
