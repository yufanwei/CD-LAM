#!/usr/bin/env python3
"""Create, validate, or install a platform-specific CD-LAM offline cache.

The cache is a transport artifact, not part of the source release. It captures
the Python packages from a known-good isolated core environment, a wheel built
from the current checkout, and the Ruff executable. The cache is intentionally
bound to one Python minor version, implementation tag, operating system, and
machine architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
CACHE_KIND = "cdlam_core_runtime_cache"
MANIFEST_NAME = "offline-cache.json"
EXCLUDED_NAMES = {
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "cd_lam",
    "cd_lam.egg-info",
}


class OfflineCacheError(RuntimeError):
    """Raised when an offline cache is incomplete or incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_summary(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    bytes_total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            digest.update(f"L\0{relative}\0{target}\0".encode())
            files += 1
            continue
        if not path.is_file():
            continue
        value = _sha256(path)
        size = path.stat().st_size
        digest.update(f"F\0{relative}\0{size}\0{value}\0".encode())
        files += 1
        bytes_total += size
    return {"files": files, "bytes": bytes_total, "sha256": digest.hexdigest()}


def _runtime_identity(python: str | None = None) -> dict[str, str]:
    if python is None:
        return {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "cache_tag": str(sys.implementation.cache_tag),
            "system": platform.system(),
            "machine": platform.machine(),
        }
    program = (
        "import json,platform,sys;"
        "print(json.dumps({'implementation':platform.python_implementation(),"
        "'version':platform.python_version(),"
        "'major_minor':f'{sys.version_info.major}.{sys.version_info.minor}',"
        "'cache_tag':str(sys.implementation.cache_tag),"
        "'system':platform.system(),'machine':platform.machine()}))"
    )
    result = subprocess.run(
        [python, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _locked_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(
        (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise OfflineCacheError(
                f"requirements.lock line {line_number} is not an exact pin: {line!r}"
            )
        name, version = line.split("==", 1)
        key = name.strip().lower().replace("_", "-")
        if not key or not version or key in result:
            raise OfflineCacheError(
                f"requirements.lock line {line_number} is invalid or duplicate"
            )
        result[key] = version
    if not result:
        raise OfflineCacheError("requirements.lock contains no pinned distributions")
    return result


def _require_locked_versions(installed: dict[str, str], locked: dict[str, str]) -> None:
    mismatches = {
        name: {"expected": expected, "observed": installed.get(name)}
        for name, expected in locked.items()
        if installed.get(name) != expected
    }
    if mismatches:
        raise OfflineCacheError(
            f"source environment does not match requirements.lock: {mismatches}"
        )


def _installed_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    locked = _locked_versions()
    result: dict[str, str] = {}
    for name in locked:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise OfflineCacheError(
                f"the source environment is missing required distribution {name!r}"
            ) from exc
    _require_locked_versions(result, locked)
    return result


def _copy_ignore(_directory: str, names: Iterable[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if (
            name in EXCLUDED_NAMES
            or name.startswith("cd_lam-")
            or name.startswith("__editable__.cd_lam-")
            or name.endswith((".pyc", ".pyo", ".pth"))
        ):
            ignored.add(name)
    return ignored


def _assert_self_contained_links(root: Path) -> None:
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise OfflineCacheError(
                f"offline cache contains a broken symlink: {path}"
            ) from exc
        if not resolved.is_relative_to(resolved_root):
            raise OfflineCacheError(
                f"offline cache symlink escapes the cache root: {path} -> {resolved}"
            )


def _normalize_permissions(root: Path) -> None:
    """Make a cache readable after transfer to another Unix account."""

    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            executable = bool(path.stat().st_mode & 0o111)
            path.chmod(0o755 if executable else 0o644)


def _load_manifest(cache: Path) -> dict[str, Any]:
    path = cache / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineCacheError(
            f"cannot read offline cache manifest {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise OfflineCacheError("offline cache manifest must contain an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise OfflineCacheError("unsupported offline cache schema")
    if payload.get("kind") != CACHE_KIND:
        raise OfflineCacheError("offline cache kind is not CD-LAM core runtime")
    return payload


def create_cache(output: Path, *, builder_python: str | None = None) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise OfflineCacheError(f"refusing to overwrite existing cache path: {output}")
    source_site = Path(sysconfig.get_paths()["purelib"]).resolve()
    if not source_site.is_dir():
        raise OfflineCacheError(
            f"source site-packages directory is missing: {source_site}"
        )
    versions = _installed_versions()
    lock_hash = _sha256(ROOT / "requirements.lock")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=str(output.parent))
    )
    try:
        site_target = temporary / "site-packages"
        shutil.copytree(
            source_site,
            site_target,
            symlinks=True,
            ignore=_copy_ignore,
        )
        _assert_self_contained_links(site_target)

        wheels = temporary / "wheels"
        wheels.mkdir()
        environment = dict(os.environ)
        environment.update({"PIP_NO_INDEX": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        builder = (
            str(Path(builder_python).expanduser().resolve())
            if builder_python
            else sys.executable
        )
        try:
            subprocess.run(
                [
                    builder,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(wheels),
                    str(ROOT),
                ],
                check=True,
                cwd=ROOT,
                env=environment,
            )
        except subprocess.CalledProcessError as exc:
            raise OfflineCacheError(
                "wheel build failed; use a builder with setuptools>=77 and wheel, "
                "or pass --builder-python"
            ) from exc
        built = sorted(wheels.glob("cd_lam-*.whl"))
        if len(built) != 1:
            raise OfflineCacheError(f"expected one CD-LAM wheel, found {len(built)}")

        source_ruff = Path(sys.prefix) / "bin" / "ruff"
        if not source_ruff.is_file():
            raise OfflineCacheError(f"Ruff executable is missing: {source_ruff}")
        bin_dir = temporary / "bin"
        bin_dir.mkdir()
        shutil.copy2(source_ruff, bin_dir / "ruff")
        _normalize_permissions(temporary)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": CACHE_KIND,
            "runtime": _runtime_identity(),
            "requirements_lock_sha256": lock_hash,
            "distributions": versions,
            "site_packages": _tree_summary(site_target),
            "wheel": {
                "path": f"wheels/{built[0].name}",
                "bytes": built[0].stat().st_size,
                "sha256": _sha256(built[0]),
            },
            "executables": {
                "ruff": {
                    "path": "bin/ruff",
                    "bytes": (bin_dir / "ruff").stat().st_size,
                    "sha256": _sha256(bin_dir / "ruff"),
                }
            },
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
        return payload
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_cache(cache: Path, *, python: str | None = None) -> dict[str, Any]:
    cache = cache.expanduser().resolve()
    payload = _load_manifest(cache)
    expected_runtime = payload.get("runtime")
    actual_runtime = _runtime_identity(python)
    identity_fields = (
        "implementation",
        "major_minor",
        "cache_tag",
        "system",
        "machine",
    )
    if not isinstance(expected_runtime, dict) or any(
        expected_runtime.get(field) != actual_runtime.get(field)
        for field in identity_fields
    ):
        raise OfflineCacheError(
            "offline cache runtime mismatch: "
            f"expected={expected_runtime!r} actual={actual_runtime!r}"
        )
    lock_hash = _sha256(ROOT / "requirements.lock")
    if payload.get("requirements_lock_sha256") != lock_hash:
        raise OfflineCacheError(
            "offline cache was built for a different requirements.lock"
        )
    raw_distributions = payload.get("distributions")
    if not isinstance(raw_distributions, dict):
        raise OfflineCacheError("offline cache distribution inventory is missing")
    distributions = {
        str(name).lower().replace("_", "-"): str(version)
        for name, version in raw_distributions.items()
    }
    _require_locked_versions(distributions, _locked_versions())

    site = cache / "site-packages"
    if not site.is_dir() or _tree_summary(site) != payload.get("site_packages"):
        raise OfflineCacheError("offline cache site-packages digest does not match")
    _assert_self_contained_links(site)

    wheel = payload.get("wheel")
    if not isinstance(wheel, dict):
        raise OfflineCacheError("offline cache wheel declaration is missing")
    wheel_path = cache / str(wheel.get("path", ""))
    if (
        not wheel_path.is_file()
        or wheel_path.stat().st_size != wheel.get("bytes")
        or _sha256(wheel_path) != wheel.get("sha256")
    ):
        raise OfflineCacheError("offline CD-LAM wheel does not match its manifest")

    executables = payload.get("executables")
    ruff = executables.get("ruff") if isinstance(executables, dict) else None
    if not isinstance(ruff, dict):
        raise OfflineCacheError("offline Ruff executable declaration is missing")
    ruff_path = cache / str(ruff.get("path", ""))
    if (
        not ruff_path.is_file()
        or ruff_path.stat().st_size != ruff.get("bytes")
        or _sha256(ruff_path) != ruff.get("sha256")
    ):
        raise OfflineCacheError("offline Ruff executable does not match its manifest")
    return payload


def _target_site(python: str) -> Path:
    program = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    result = subprocess.run(
        [python, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _isolated_venv_python(target_python: str) -> Path:
    """Return a lexical venv interpreter path after checking isolation.

    Virtual environments normally expose ``bin/python`` as a symlink to the
    base interpreter. Resolving that symlink before locating ``pyvenv.cfg``
    would inspect the base environment and incorrectly reject a valid venv.
    """

    target = Path(os.path.abspath(Path(target_python).expanduser()))
    if not target.is_file():
        raise OfflineCacheError(f"target Python is missing: {target}")
    cfg = target.parents[1] / "pyvenv.cfg"
    text = cfg.read_text(encoding="utf-8") if cfg.is_file() else ""
    if "include-system-site-packages = false" not in text.lower():
        raise OfflineCacheError(
            f"target must be an isolated virtual environment: {target}"
        )
    return target


def _replace_site_packages(source: Path, target: Path, venv_root: Path) -> None:
    """Atomically replace an isolated venv's package tree with the cache."""

    source = source.resolve()
    target = target.resolve()
    venv_root = venv_root.resolve()
    if not source.is_dir():
        raise OfflineCacheError(f"cached site-packages directory is missing: {source}")
    if not target.is_relative_to(venv_root):
        raise OfflineCacheError(
            f"refusing to replace site-packages outside target venv: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.cdlam-tmp-{os.getpid()}")
    backup = target.with_name(f".{target.name}.cdlam-backup-{os.getpid()}")
    if temporary.exists() or backup.exists():
        raise OfflineCacheError("stale offline-cache installation directory exists")
    moved_existing = False
    try:
        shutil.copytree(source, temporary, symlinks=True)
        _assert_self_contained_links(temporary)
        if target.exists():
            target.rename(backup)
            moved_existing = True
        temporary.rename(target)
        if moved_existing:
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if moved_existing and backup.exists() and not target.exists():
            backup.rename(target)
        raise


def install_cache(cache: Path, target_python: str) -> dict[str, Any]:
    target = _isolated_venv_python(target_python)
    payload = validate_cache(cache, python=str(target))
    site = _target_site(str(target))
    _replace_site_packages(
        cache / "site-packages",
        site,
        target.parents[1],
    )
    source_ruff = cache / payload["executables"]["ruff"]["path"]
    target_ruff = target.parent / "ruff"
    shutil.copy2(source_ruff, target_ruff)
    target_ruff.chmod(
        target_ruff.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    wheel = cache / payload["wheel"]["path"]
    environment = dict(os.environ)
    environment.update({"PIP_NO_INDEX": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    subprocess.run(
        [
            str(target),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        check=True,
        env=environment,
    )
    subprocess.run([str(target), "-m", "pip", "check"], check=True, env=environment)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create", help="Create a cache from this Python environment."
    )
    create.add_argument("--output", type=Path, required=True)
    create.add_argument(
        "--builder-python",
        help="Interpreter with build, wheel, and setuptools>=77; defaults to this interpreter.",
    )
    validate = subparsers.add_parser(
        "validate", help="Validate cache contents and compatibility."
    )
    validate.add_argument("--cache", type=Path, required=True)
    install = subparsers.add_parser(
        "install", help="Install a cache into a fresh isolated venv."
    )
    install.add_argument("--cache", type=Path, required=True)
    install.add_argument("--target-python", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "create":
        payload = create_cache(args.output, builder_python=args.builder_python)
    elif args.command == "validate":
        payload = validate_cache(args.cache)
    else:
        payload = install_cache(args.cache, args.target_python)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OfflineCacheError as exc:
        raise SystemExit(f"offline_cache: {exc}") from exc
