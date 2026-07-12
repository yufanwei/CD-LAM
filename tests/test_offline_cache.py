from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offline_cache.py"
SPEC = importlib.util.spec_from_file_location("cdlam_offline_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
OFFLINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OFFLINE)


def _write_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    site = cache / "site-packages"
    wheels = cache / "wheels"
    binaries = cache / "bin"
    site.mkdir(parents=True)
    wheels.mkdir()
    binaries.mkdir()
    (site / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    wheel = wheels / "cd_lam-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-fixture")
    ruff = binaries / "ruff"
    ruff.write_bytes(b"ruff-fixture")
    payload = {
        "schema_version": OFFLINE.SCHEMA_VERSION,
        "kind": OFFLINE.CACHE_KIND,
        "runtime": OFFLINE._runtime_identity(),
        "requirements_lock_sha256": OFFLINE._sha256(ROOT / "requirements.lock"),
        "distributions": OFFLINE._locked_versions(),
        "site_packages": OFFLINE._tree_summary(site),
        "wheel": {
            "path": str(wheel.relative_to(cache)),
            "bytes": wheel.stat().st_size,
            "sha256": OFFLINE._sha256(wheel),
        },
        "executables": {
            "ruff": {
                "path": str(ruff.relative_to(cache)),
                "bytes": ruff.stat().st_size,
                "sha256": OFFLINE._sha256(ruff),
            }
        },
    }
    (cache / OFFLINE.MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return cache


def test_offline_cache_validates_complete_platform_bound_fixture(
    tmp_path: Path,
) -> None:
    cache = _write_cache(tmp_path)

    payload = OFFLINE.validate_cache(cache)

    assert payload["kind"] == OFFLINE.CACHE_KIND
    assert payload["site_packages"]["files"] == 1


def test_lock_file_contains_exact_build_tool_pins() -> None:
    locked = OFFLINE._locked_versions()

    assert locked["build"] == "1.3.0"
    assert locked["setuptools"] == "83.0.0"
    assert locked["wheel"] == "0.45.1"


def test_cache_source_must_match_every_locked_version() -> None:
    with pytest.raises(OFFLINE.OfflineCacheError, match="does not match"):
        OFFLINE._require_locked_versions(
            {"build": "1.5.1", "wheel": "0.45.1"},
            {"build": "1.3.0", "wheel": "0.45.1"},
        )


def test_cache_manifest_must_match_every_locked_version(tmp_path: Path) -> None:
    cache = _write_cache(tmp_path)
    manifest_path = cache / OFFLINE.MANIFEST_NAME
    payload = json.loads(manifest_path.read_text())
    payload["distributions"]["build"] = "1.5.1"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OFFLINE.OfflineCacheError, match="does not match"):
        OFFLINE.validate_cache(cache)


def test_offline_cache_detects_package_tree_tampering(tmp_path: Path) -> None:
    cache = _write_cache(tmp_path)
    (cache / "site-packages" / "example.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(OFFLINE.OfflineCacheError, match="site-packages digest"):
        OFFLINE.validate_cache(cache)


def test_offline_cache_rejects_runtime_mismatch(tmp_path: Path) -> None:
    cache = _write_cache(tmp_path)
    manifest_path = cache / OFFLINE.MANIFEST_NAME
    payload = json.loads(manifest_path.read_text())
    payload["runtime"]["machine"] = "incompatible-machine"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OFFLINE.OfflineCacheError, match="runtime mismatch"):
        OFFLINE.validate_cache(cache)


def test_cache_copy_filter_removes_external_path_injection_files() -> None:
    ignored = OFFLINE._copy_ignore(
        "unused",
        [
            "dependency.pth",
            "__editable__.cd_lam-0.1.0.pth",
            "cd_lam",
            "cd_lam-0.1.0.dist-info",
            "safe_package",
        ],
    )

    assert ignored == {
        "dependency.pth",
        "__editable__.cd_lam-0.1.0.pth",
        "cd_lam",
        "cd_lam-0.1.0.dist-info",
    }


def test_cache_creation_refuses_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()

    with pytest.raises(OFFLINE.OfflineCacheError, match="refusing to overwrite"):
        OFFLINE.create_cache(target)


def test_virtualenv_python_symlink_is_checked_at_venv_location(
    tmp_path: Path,
) -> None:
    base_python = tmp_path / "base" / "python3"
    base_python.parent.mkdir()
    base_python.write_bytes(b"python-fixture")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    target_python = venv / "bin" / "python"
    target_python.symlink_to(base_python)

    target = OFFLINE._isolated_venv_python(str(target_python))

    assert target == target_python
    assert target.resolve() == base_python


def test_cached_site_packages_replace_venv_bootstrap_packages(tmp_path: Path) -> None:
    source = tmp_path / "cache" / "site-packages"
    source.mkdir(parents=True)
    (source / "setuptools-83.0.0.dist-info").mkdir()
    venv = tmp_path / "venv"
    target = venv / "lib" / "python3.10" / "site-packages"
    target.mkdir(parents=True)
    (target / "setuptools-65.5.0.dist-info").mkdir()

    OFFLINE._replace_site_packages(source, target, venv)

    assert (target / "setuptools-83.0.0.dist-info").is_dir()
    assert not (target / "setuptools-65.5.0.dist-info").exists()


def test_site_package_replacement_cannot_escape_target_venv(tmp_path: Path) -> None:
    source = tmp_path / "cache" / "site-packages"
    source.mkdir(parents=True)
    outside = tmp_path / "outside" / "site-packages"
    with pytest.raises(OFFLINE.OfflineCacheError, match="outside target venv"):
        OFFLINE._replace_site_packages(source, outside, tmp_path / "venv")


def test_bootstrap_requires_explicit_system_runtime_reuse() -> None:
    text = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")

    assert '"$PYTHON" -m venv "$VENV"' in text
    assert '"$PYTHON" -m venv --system-site-packages "$VENV"' in text
    assert '[[ "$REUSE_SYSTEM_RUNTIME" == 1 ]]' in text
