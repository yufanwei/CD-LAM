#!/usr/bin/env python3
"""Materialize the CD-LAM ACWM runtime from a pinned base and bundled overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERLAY_ROOT = BUNDLE_ROOT / "third_party" / "acwm_overlay"
DEFAULT_MANIFEST = DEFAULT_OVERLAY_ROOT / "manifest.json"
PROVENANCE_NAME = ".cdlam-runtime-source.json"
RUNTIME_TREE_ALGORITHM = "path-size-sha256-v1"


class RuntimeSourceError(RuntimeError):
    """Raised when the cached runtime source does not match the release manifest."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_tree_summary(root: Path) -> dict[str, Any]:
    """Hash every stable runtime file, including unmodified upstream sources."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeSourceError(f"runtime tree is missing: {root}")
    digest = hashlib.sha256()
    files = 0
    bytes_total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if relative == Path(PROVENANCE_NAME):
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise RuntimeSourceError(f"runtime tree contains a link: {relative}")
        if not path.is_file():
            continue
        value = sha256(path)
        size = path.stat().st_size
        digest.update(f"F\0{relative.as_posix()}\0{size}\0{value}\0".encode("utf-8"))
        files += 1
        bytes_total += size
    return {
        "algorithm": RUNTIME_TREE_ALGORITHM,
        "files": files,
        "bytes": bytes_total,
        "sha256": digest.hexdigest(),
    }


def _expected_runtime_tree(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("runtime_tree")
    if not isinstance(value, dict):
        raise RuntimeSourceError(
            "runtime manifest has no complete runtime_tree binding"
        )
    if value.get("algorithm") != RUNTIME_TREE_ALGORITHM:
        raise RuntimeSourceError("runtime manifest uses an unsupported tree algorithm")
    files = value.get("files")
    bytes_total = value.get("bytes")
    tree_hash = value.get("sha256")
    if (
        isinstance(files, bool)
        or not isinstance(files, int)
        or files < 1
        or isinstance(bytes_total, bool)
        or not isinstance(bytes_total, int)
        or bytes_total < 1
        or not isinstance(tree_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", tree_hash) is None
    ):
        raise RuntimeSourceError("runtime manifest runtime_tree is invalid")
    return {
        "algorithm": RUNTIME_TREE_ALGORITHM,
        "files": files,
        "bytes": bytes_total,
        "sha256": tree_hash,
    }


def safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeSourceError(f"{label} must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeSourceError(f"{label} is not a safe relative path: {value!r}")
    return Path(*path.parts)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSourceError(f"cannot read runtime manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeSourceError("unsupported runtime source manifest")
    if payload.get("runtime_id") != "cdlam-acwm-runtime":
        raise RuntimeSourceError("runtime manifest has an unexpected runtime_id")
    if payload.get("publication_status") != "bundled":
        raise RuntimeSourceError("runtime overlay is not marked as bundled")
    if payload.get("base_repository") != "https://github.com/NVIDIA/DreamDojo.git":
        raise RuntimeSourceError("runtime base_repository is not the pinned upstream")
    commit = payload.get("base_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeSourceError(
            "runtime base_commit must contain 40 hexadecimal characters"
        )
    rows = payload.get("overlays")
    if not isinstance(rows, list) or not rows:
        raise RuntimeSourceError("runtime manifest must contain overlay rows")
    _expected_runtime_tree(payload)
    return payload


def _verify_overlay(overlay_root: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    relative = safe_relative(row.get("path"), "overlay.path")
    source = overlay_root / relative
    if not source.is_file():
        raise RuntimeSourceError(f"overlay file is missing: {source}")
    if source.stat().st_size != row.get("bytes") or sha256(source) != row.get("sha256"):
        raise RuntimeSourceError(f"overlay file does not match its manifest: {source}")
    operation = row.get("operation")
    if operation not in {"added", "modified"}:
        raise RuntimeSourceError(f"overlay operation is invalid for {relative}")
    if operation == "modified":
        base_digest = row.get("base_sha256")
        if (
            not isinstance(base_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", base_digest) is None
        ):
            raise RuntimeSourceError(f"base SHA-256 is missing for {relative}")
    return source, relative


def _verify_base_relation(root: Path, relative: Path, row: dict[str, Any]) -> None:
    destination = root / relative
    if row["operation"] == "added":
        if destination.exists():
            raise RuntimeSourceError(
                f"overlay marks an existing base path as added: {relative}"
            )
        return
    if not destination.is_file():
        raise RuntimeSourceError(f"modified base path is missing: {relative}")
    if sha256(destination) != row["base_sha256"]:
        raise RuntimeSourceError(f"modified base path hash drifted: {relative}")


def _safe_extract(
    archive: Path,
    output: Path,
    excluded_links: set[str],
) -> None:
    with tarfile.open(archive) as handle:
        members = []
        for member in handle.getmembers():
            relative = safe_relative(member.name, "archive member")
            destination = (output / relative).resolve()
            if not destination.is_relative_to(output.resolve()):
                raise RuntimeSourceError(
                    f"archive member escapes output: {member.name}"
                )
            if member.issym() or member.islnk():
                if relative.as_posix() in excluded_links:
                    continue
                raise RuntimeSourceError(
                    f"runtime base archive must not contain links: {member.name}"
                )
            members.append(member)
        handle.extractall(output, members=members)


def verify_runtime(
    output: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    payload = load_manifest(manifest_path)
    provenance_path = output / PROVENANCE_NAME
    if not output.is_dir() or not provenance_path.is_file():
        raise RuntimeSourceError(f"staged runtime provenance is missing: {output}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSourceError(f"invalid staged runtime provenance: {exc}") from exc
    expected = {
        "runtime_id": payload["runtime_id"],
        "base_repository": payload["base_repository"],
        "base_commit": payload["base_commit"],
        "overlay_files": len(payload["overlays"]),
        "overlay_manifest_sha256": sha256(manifest_path),
        "runtime_tree": _expected_runtime_tree(payload),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RuntimeSourceError(f"staged runtime provenance mismatch: {key}")
    for row in payload["overlays"]:
        relative = safe_relative(row.get("path"), "overlay.path")
        destination = output / relative
        if (
            not destination.is_file()
            or destination.stat().st_size != row.get("bytes")
            or sha256(destination) != row.get("sha256")
        ):
            raise RuntimeSourceError(f"staged runtime file drifted: {relative}")
    for value in payload.get("required_runtime_paths", []):
        relative = safe_relative(value, "required_runtime_paths")
        if not (output / relative).exists():
            raise RuntimeSourceError(f"staged runtime path is missing: {relative}")
    observed_tree = runtime_tree_summary(output)
    if observed_tree != expected["runtime_tree"]:
        raise RuntimeSourceError(
            "staged runtime tree drifted from the pinned base-plus-overlay manifest: "
            f"expected={expected['runtime_tree']!r} observed={observed_tree!r}"
        )
    return provenance


def stage_runtime(
    base_git: Path,
    overlay_root: Path,
    output: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    base_git = base_git.expanduser().resolve()
    overlay_root = overlay_root.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if output.exists():
        raise RuntimeSourceError(f"refusing to overwrite runtime path: {output}")
    if not (base_git / ".git").exists():
        raise RuntimeSourceError(f"base source is not a Git checkout: {base_git}")
    payload = load_manifest(manifest_path)
    commit = payload["base_commit"]
    try:
        subprocess.run(
            ["git", "-C", str(base_git), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeSourceError(
            f"base source does not contain required commit {commit}"
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=str(output.parent))
    )
    archive = temporary.parent / f".{output.name}.base.{temporary.name}.tar"
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(base_git),
                "archive",
                "--format=tar",
                "-o",
                str(archive),
                commit,
            ],
            check=True,
        )
        excluded_links = {
            safe_relative(value, "excluded_base_paths").as_posix()
            for value in payload.get("excluded_base_paths", [])
        }
        _safe_extract(archive, temporary, excluded_links)
        copied: list[dict[str, Any]] = []
        for row in payload.get("overlays", []):
            if not isinstance(row, dict):
                raise RuntimeSourceError("overlay manifest rows must be objects")
            source, relative = _verify_overlay(overlay_root, row)
            _verify_base_relation(temporary, relative, row)
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(dict(row))
        for value in payload.get("required_runtime_paths", []):
            relative = safe_relative(value, "required_runtime_paths")
            if not (temporary / relative).exists():
                raise RuntimeSourceError(
                    f"base commit is missing required runtime path: {relative}"
                )
        observed_tree = runtime_tree_summary(temporary)
        expected_tree = _expected_runtime_tree(payload)
        if observed_tree != expected_tree:
            raise RuntimeSourceError(
                "staged runtime tree does not match the pinned base-plus-overlay manifest: "
                f"expected={expected_tree!r} observed={observed_tree!r}"
            )
        provenance = {
            "schema_version": 1,
            "runtime_id": payload.get("runtime_id"),
            "base_repository": payload.get("base_repository"),
            "base_commit": commit,
            "overlay_files": len(copied),
            "overlay_manifest_sha256": sha256(manifest_path),
            "runtime_tree": expected_tree,
            "overlays": copied,
        }
        (temporary / PROVENANCE_NAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
        return provenance
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        archive.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-git", type=Path)
    parser.add_argument("--overlay-root", type=Path, default=DEFAULT_OVERLAY_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        provenance = verify_runtime(args.output, args.manifest)
    else:
        if args.base_git is None:
            parser.error("--base-git is required unless --verify-existing is used")
        provenance = stage_runtime(
            args.base_git,
            args.overlay_root,
            args.output,
            args.manifest,
        )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeSourceError as exc:
        raise SystemExit(f"stage_acwm_runtime: {exc}") from exc
