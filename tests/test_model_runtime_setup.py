from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCTOR_PATH = ROOT / "scripts/model_runtime_doctor.py"
STAGE_PATH = ROOT / "internal/tools/stage_acwm_runtime.py"
SPEC = importlib.util.spec_from_file_location("model_runtime_doctor", DOCTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)
STAGE_SPEC = importlib.util.spec_from_file_location("stage_acwm_runtime", STAGE_PATH)
assert STAGE_SPEC is not None and STAGE_SPEC.loader is not None
stage = importlib.util.module_from_spec(STAGE_SPEC)
STAGE_SPEC.loader.exec_module(stage)


def _lock() -> dict[str, object]:
    return doctor.load_lock(ROOT / "configs/model_runtime.lock.json")


def test_model_runtime_lock_records_proven_stack() -> None:
    lock = _lock()
    extra_lock = (ROOT / "configs/model_runtime.extra.lock.txt").read_text(
        encoding="utf-8"
    )
    assert lock["profile"] == "linux-x86_64-cpython310-cu128-torch27"
    assert lock["installer"] == {
        "uv_version": "0.9.7",
        "upstream_extra": "cu128",
        "include_upstream_dev_group": False,
        "post_sync_reinstall": {"opencv-python-headless": "4.11.0.86"},
    }
    assert lock["source"]["revision"] == ("02f119b759d5c7f84a399fdeea3c6e82e7ed6cff")
    assert lock["critical_distributions"]["torch"] == "2.7.0+cu128"
    assert lock["critical_distributions"]["transformer-engine"] == ("2.2+cu128.torch27")
    assert lock["critical_distributions"]["packaging"] == "25.0"
    assert lock["critical_distributions"]["opencv-python"] == "4.11.0.86"
    assert lock["critical_distributions"]["opencv-python-headless"] == "4.11.0.86"
    assert lock["critical_distributions"]["h5py"] == "3.16.0"
    assert extra_lock.splitlines().count("h5py==3.16.0") == 1
    assert lock["critical_distributions"]["lightning"] == "2.6.5"
    assert lock["critical_distributions"]["pytorch-lightning"] == "2.6.5"
    assert "groot_dreams.dataloader" in lock["required_modules"]
    assert "h5py" in lock["required_modules"]
    assert "lightning" in lock["required_modules"]
    assert {
        (
            row["requiring_distribution"],
            row["required_distribution"],
            row["specifier"],
            row["installed_version"],
        )
        for row in lock["allowed_metadata_conflicts"]
    } == {
        ("cosmos-predict2", "cosmos-oss", "==0.1.0", "1.4.1"),
        ("megatron-core", "numpy", "<2.0.0", "2.2.6"),
    }
    assert lock["recorded_h100_execution"]["nvidia_driver"] == "580.126.09"
    assert lock["platform"]["nvidia_driver_minimum"] == "570.124.06"


def test_source_contract_checks_lock_and_overlay_hashes(tmp_path: Path) -> None:
    lock = copy.deepcopy(_lock())
    acwm = tmp_path / "acwm-runtime"
    acwm.mkdir()
    uv_lock = acwm / "uv.lock"
    uv_lock.write_text("locked\n", encoding="utf-8")
    lock["source"]["uv_lock_sha256"] = hashlib.sha256(b"locked\n").hexdigest()
    for relative in (
        "pyproject.toml",
        "packages/cosmos-oss/pyproject.toml",
        "packages/cosmos-cuda/pyproject.toml",
        "cosmos_predict2/__about__.py",
    ):
        path = acwm / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("present\n", encoding="utf-8")

    overlay = tmp_path / "manifest.json"
    overlay.write_text(
        json.dumps(
            {
                "base_commit": lock["source"]["revision"],
                "base_repository": lock["source"]["repository"],
            }
        ),
        encoding="utf-8",
    )
    (acwm / ".cdlam-runtime-source.json").write_text(
        json.dumps(
            {
                "base_commit": lock["source"]["revision"],
                "base_repository": lock["source"]["repository"],
                "overlay_manifest_sha256": doctor.sha256_file(overlay),
            }
        ),
        encoding="utf-8",
    )

    assert doctor.source_errors(lock, acwm, overlay) == []
    uv_lock.write_text("drifted\n", encoding="utf-8")
    errors = doctor.source_errors(lock, acwm, overlay)
    assert any("uv.lock SHA-256 mismatch" in error for error in errors)


def test_doctor_rejects_drift_in_unoverlaid_runtime_file(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    added = runtime / "added.py"
    added.write_text("overlay = True\n", encoding="utf-8")
    stable = runtime / "stable.py"
    stable.write_text("stable = True\n", encoding="utf-8")
    tree = stage.runtime_tree_summary(runtime)

    manifest = tmp_path / "manifest.json"
    document = {
        "schema_version": 1,
        "runtime_id": "cdlam-acwm-runtime",
        "publication_status": "bundled",
        "base_repository": "https://github.com/NVIDIA/DreamDojo.git",
        "base_commit": "0" * 40,
        "runtime_tree": tree,
        "required_runtime_paths": ["added.py", "stable.py"],
        "overlays": [
            {
                "path": "added.py",
                "bytes": added.stat().st_size,
                "sha256": stage.sha256(added),
                "operation": "added",
            }
        ],
    }
    manifest.write_text(json.dumps(document), encoding="utf-8")
    (runtime / ".cdlam-runtime-source.json").write_text(
        json.dumps(
            {
                "runtime_id": document["runtime_id"],
                "base_repository": document["base_repository"],
                "base_commit": document["base_commit"],
                "overlay_files": 1,
                "overlay_manifest_sha256": stage.sha256(manifest),
                "runtime_tree": tree,
            }
        ),
        encoding="utf-8",
    )

    assert doctor.runtime_tree_errors(runtime, manifest, STAGE_PATH) == []
    stable.write_text("stable = False\n", encoding="utf-8")
    errors = doctor.runtime_tree_errors(runtime, manifest, STAGE_PATH)
    assert any("runtime tree drifted" in error for error in errors)


def test_probe_contract_rejects_wrong_package_and_source(tmp_path: Path) -> None:
    lock = _lock()
    acwm = tmp_path / "acwm"
    versions = dict(lock["critical_distributions"])
    module_imports = {
        name: {"ok": True, "error": None} for name in lock["required_modules"]
    }
    direct_urls = {
        name: {"url": (acwm / relative).resolve().as_uri()}
        for name, relative in lock["editable_sources"].items()
    }
    direct_urls["pytorch3d"] = {
        "url": lock["extra_sources"]["pytorch3d"]["repository"],
        "vcs_info": {"commit_id": lock["extra_sources"]["pytorch3d"]["revision"]},
    }
    payload = {
        "python": "3.10.18",
        "implementation": "CPython",
        "system": "Linux",
        "machine": "x86_64",
        "libc": "2.35",
        "versions": versions,
        "module_imports": module_imports,
        "direct_urls": direct_urls,
        "metadata_conflicts": [
            {field: row[field] for field in doctor._CONFLICT_FIELDS}
            for row in lock["allowed_metadata_conflicts"]
        ],
        "metadata_errors": [],
        "duplicate_distributions": [],
    }
    assert doctor.probe_errors(lock, payload, acwm) == []

    payload["versions"]["torch"] = "2.7.1+cu128"
    payload["direct_urls"]["cosmos-predict2"] = {
        "url": (tmp_path / "wrong-source").resolve().as_uri()
    }
    errors = doctor.probe_errors(lock, payload, acwm)
    assert any("distribution torch mismatch" in error for error in errors)
    assert any(
        "cosmos-predict2 is not installed from staged source" in error
        for error in errors
    )


def test_metadata_conflict_allowlist_is_exact(tmp_path: Path) -> None:
    lock = _lock()
    expected = [
        {field: row[field] for field in doctor._CONFLICT_FIELDS}
        for row in lock["allowed_metadata_conflicts"]
    ]
    payload = {
        "metadata_conflicts": copy.deepcopy(expected),
        "metadata_errors": [],
        "duplicate_distributions": [],
    }
    assert doctor.metadata_conflict_errors(lock, payload) == []

    payload["metadata_conflicts"].append(
        {
            "kind": "missing",
            "requiring_distribution": "example",
            "requiring_version": "1.0",
            "required_distribution": "absent",
            "specifier": ">=1",
            "installed_version": None,
        }
    )
    errors = doctor.metadata_conflict_errors(lock, payload)
    assert any("unexpected dependency metadata conflict" in error for error in errors)

    payload["metadata_conflicts"] = expected[1:]
    errors = doctor.metadata_conflict_errors(lock, payload)
    assert any(
        "expected dependency metadata conflict is missing" in error for error in errors
    )

    payload["metadata_conflicts"] = expected
    payload["metadata_errors"] = ["bad requirement"]
    payload["duplicate_distributions"] = ["numpy"]
    errors = doctor.metadata_conflict_errors(lock, payload)
    assert "dependency metadata error: bad requirement" in errors
    assert "duplicate installed distribution: numpy" in errors


def test_required_module_check_uses_real_import_results(tmp_path: Path) -> None:
    lock = _lock()
    acwm = tmp_path / "acwm"
    payload = {
        "python": "3.10.18",
        "implementation": "CPython",
        "system": "Linux",
        "machine": "x86_64",
        "libc": "2.35",
        "versions": dict(lock["critical_distributions"]),
        "module_imports": {
            name: {"ok": True, "error": None} for name in lock["required_modules"]
        },
        "direct_urls": {
            name: {"url": (acwm / relative).resolve().as_uri()}
            for name, relative in lock["editable_sources"].items()
        },
        "metadata_conflicts": [
            {field: row[field] for field in doctor._CONFLICT_FIELDS}
            for row in lock["allowed_metadata_conflicts"]
        ],
        "metadata_errors": [],
        "duplicate_distributions": [],
    }
    payload["direct_urls"]["pytorch3d"] = {
        "vcs_info": {"commit_id": lock["extra_sources"]["pytorch3d"]["revision"]}
    }
    payload["module_imports"]["decord"] = {
        "ok": False,
        "error": "ImportError: incompatible binary",
    }
    errors = doctor.probe_errors(lock, payload, acwm)
    assert any(
        "required module import failed: decord: ImportError" in error
        for error in errors
    )


def test_bootstrap_dry_run_is_non_mutating(tmp_path: Path) -> None:
    environment = tmp_path / ".venv"
    dependencies = tmp_path / "deps"
    tool_bin = tmp_path / "tool-bin"
    tool_bin.mkdir()
    dirname = shutil.which("dirname")
    assert dirname is not None
    (tool_bin / "dirname").symlink_to(dirname)
    clean_environment = os.environ.copy()
    clean_environment["PATH"] = str(tool_bin)
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "scripts/bootstrap_model_runtime.sh"),
            "--dry-run",
            "--python",
            sys.executable,
            "--environment",
            str(environment),
            "--deps-root",
            str(dependencies),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=clean_environment,
    )
    if sys.version_info[:2] != (3, 10):
        assert result.returncode == 2
        assert "requires CPython 3.10" in result.stderr
        assert not environment.exists()
        assert not dependencies.exists()
        return
    assert result.returncode == 0, result.stderr
    assert "uv 0.9.7 sync --locked --no-dev --extra cu128" in result.stdout
    assert "install locked runtime supplements" in result.stdout
    assert "reinstall opencv-python-headless==4.11.0.86" in result.stdout
    assert not environment.exists()
    assert not dependencies.exists()
    script = (ROOT / "scripts/bootstrap_model_runtime.sh").read_text(encoding="utf-8")
    assert "CDLAM_MODEL_HTTP_TIMEOUT:-300" in script
    assert 'UV_HTTP_TIMEOUT="$HTTP_TIMEOUT"' in script
    assert "pip check" not in script
    assert "model_runtime_doctor.py" in script
    assert '"opencv-python-headless==$HEADLESS_OPENCV_VERSION"' in script


def test_doctor_help_does_not_probe_environment() -> None:
    result = subprocess.run(
        [sys.executable, str(DOCTOR_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--check-driver" in result.stdout
