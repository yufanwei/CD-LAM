#!/usr/bin/env python3
"""Static release checks that do not need model weights or a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".deps",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "outputs",
}
TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".patch",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {".gitattributes", ".gitignore", "LICENSE", "Makefile", "NOTICE"}
FORBIDDEN_BINARY_SUFFIXES = {".ckpt", ".npz", ".parquet", ".pt", ".pth", ".safetensors"}
CJK_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LEGACY_VERSION_COMPONENT = re.compile(r"^V[0-9]+(?:_[0-9]+)?$", re.IGNORECASE)
PUBLIC_VERSION_LABEL = re.compile(
    r"(?:^|[^a-z0-9_])v[0-9]+(?:[._][0-9]+)*(?:[^a-z0-9_]|$)", re.IGNORECASE
)
PROJECT_RUNTIME_VERSION_LABEL = re.compile(
    r"(?<![a-z0-9])v[0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE
)
PUBLIC_HOST_MODE = re.compile(r"(?:^|[^a-z0-9_])cpu(?:[^a-z0-9_]|$)", re.IGNORECASE)
FORBIDDEN = {
    "private workspace path": re.compile(r"/(?:workspace|home)/yufan|/tmp/data"),
    "private cluster path": re.compile(r"/(?:mnt|scratch|gpfs|lustre)(?:/|-)"),
    "dated temporary tool": re.compile(r"tmp_remote_eval_tools_\\d+"),
    "private runtime namespace": re.compile(r"DREAMDOJO_ROOT|DREAMDOJO_REPO"),
    "credential material": re.compile(
        r"(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)"
    ),
}
ALLOWED_TOP_LEVEL = {
    ".gitattributes",
    ".github",
    ".gitignore",
    "LICENSE",
    "Makefile",
    "NOTICE",
    "README.md",
    "run.sh",
    "setup.sh",
    "configs",
    "docs",
    "internal",
    "pyproject.toml",
    "requirements.lock",
    "scripts",
    "src",
    "tests",
    "third_party",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fallback_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(
            part in SKIP_DIRS or part.endswith(".egg-info")
            for part in path.relative_to(ROOT).parts
        ):
            continue
        if path.is_file() or path.is_symlink():
            files.append(path)
    return files


def tracked_files() -> list[Path]:
    """Return tracked and untracked release files, with an archive fallback."""

    top_level = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        top_level.returncode
        or Path(top_level.stdout.strip()).resolve() != ROOT.resolve()
    ):
        return _fallback_files()
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        capture_output=True,
        check=True,
    )
    files: list[Path] = []
    for raw_path in proc.stdout.split(b"\0"):
        if not raw_path:
            continue
        try:
            relative = Path(raw_path.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise RuntimeError("tracked path is not valid UTF-8") from exc
        path = ROOT / relative
        if path.exists() or path.is_symlink():
            files.append(path)
    return files


def _is_release_text(path: Path) -> bool:
    return (
        path.suffix.lower() in TEXT_SUFFIXES
        or path.name in TEXT_FILENAMES
        or path.name.endswith(".env.example")
    )


def _allows_internal_reference(label: str, relative: Path) -> bool:
    """Allow declared prepared-workspace references without relaxing secrets."""

    if label not in {"private workspace path", "private runtime namespace"}:
        return False
    allowed: dict[str, set[Path]] = {
        "private workspace path": {
            Path("configs/runtime.184.env.example"),
            Path("configs/runtime.workspace.env.example"),
            Path("docs/14B_STATUS.md"),
            Path("docs/GPU_SMOKE.md"),
            Path("docs/INTERNAL_USE.md"),
            Path("docs/validation/LOCAL_GPU.md"),
            Path("docs/validation/REMOTE_184.md"),
            Path("internal/hf_release/README.md"),
            Path("internal/runtime/entries/stage1_lam.py"),
            Path("internal/runtime/entries/stage2_wm.py"),
            Path("internal/tools/sanitize_checkpoint.py"),
            Path("scripts/gpu_smoke.sh"),
            Path("tests/test_sanitize_checkpoint.py"),
        },
        "private runtime namespace": {
            Path("internal/runtime/entries/stage1_lam.py"),
            Path("internal/runtime/entries/stage2_wm.py"),
            Path("internal/runtime/entries/stage3_posttrain.py"),
            Path("scripts/run_internal.sh"),
        },
    }
    return relative in allowed[label]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    text_files = 0
    cjk_hits = 0
    files = tracked_files()

    unexpected_top_level = sorted(
        {
            path.relative_to(ROOT).parts[0]
            for path in files
            if path.relative_to(ROOT).parts[0] not in ALLOWED_TOP_LEVEL
        }
    )
    for item in unexpected_top_level:
        errors.append(f"unexpected top-level release item: {item}")

    if (ROOT / ("CITATION" + ".cff")).exists():
        errors.append("software citation metadata must not replace the paper BibTeX")

    legacy_overlay_paths = sorted(
        str(path.relative_to(ROOT))
        for path in files
        if Path("third_party/acwm_overlay") in path.relative_to(ROOT).parents
        and any(
            LEGACY_VERSION_COMPONENT.fullmatch(part)
            for part in path.relative_to(ROOT).parts
        )
    )
    if legacy_overlay_paths:
        errors.append(
            "runtime overlay contains version-named modules: "
            + ", ".join(legacy_overlay_paths[:5])
        )

    project_runtime_roots = (
        ROOT / "third_party" / "acwm_overlay" / "cdlam_integration",
        ROOT / "src" / "cdlam_runtime",
        ROOT / "internal" / "runtime",
        ROOT / "internal" / "vendor" / "scale_support",
    )
    for runtime_root in project_runtime_roots:
        if not runtime_root.is_dir():
            continue
        for candidate in runtime_root.rglob("*"):
            if not candidate.is_file() or not _is_release_text(candidate):
                continue
            text = candidate.read_text(encoding="utf-8").replace("imageio.v3", "")
            if PROJECT_RUNTIME_VERSION_LABEL.search(text):
                errors.append(
                    "project runtime contains a legacy version label: "
                    f"{candidate.relative_to(ROOT)}"
                )

    public_surface_paths = [
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "configs",
        ROOT / ".github" / "workflows",
        ROOT / "setup.sh",
        ROOT / "run.sh",
        ROOT / "Makefile",
        ROOT / "scripts" / "bootstrap.sh",
        ROOT / "scripts" / "run.sh",
        ROOT / "src" / "cd_lam" / "__main__.py",
        ROOT / "src" / "cd_lam" / "config.py",
        ROOT / "src" / "cd_lam" / "plans.py",
    ]
    for surface in public_surface_paths:
        candidates = [surface] if surface.is_file() else list(surface.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file() or not _is_release_text(candidate):
                continue
            text = candidate.read_text(encoding="utf-8")
            relative = candidate.relative_to(ROOT)
            if PUBLIC_VERSION_LABEL.search(text):
                errors.append(f"public surface contains a version label: {relative}")
            if PUBLIC_HOST_MODE.search(text) or "whl/" + "cpu" in text.lower():
                errors.append(f"public surface advertises a non-CUDA mode: {relative}")

    for path in files:
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"symlink is not release-safe: {rel}")
            continue
        size = path.stat().st_size
        if size > 20 * 1024 * 1024:
            errors.append(f"file exceeds 20 MiB: {rel} ({size} bytes)")
        if not os.access(path, os.R_OK):
            errors.append(f"file is not readable: {rel}")
        if path.suffix.lower() in FORBIDDEN_BINARY_SUFFIXES:
            errors.append(
                f"binary data/model artifact is not release-safe: {rel}; "
                "publish it through the documented external asset flow"
            )
        if not _is_release_text(path):
            continue
        text_files += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"release text is not UTF-8: {rel}: {exc}")
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if CJK_TEXT.search(line):
                cjk_hits += 1
                errors.append(f"CJK text is not release-safe: {rel}:{line_number}")
        # This file intentionally contains the patterns it enforces.
        if rel == Path("scripts/release_check.py"):
            continue
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text) and not _allows_internal_reference(label, rel):
                errors.append(f"{label}: {rel}")

    for path in files:
        rel = path.relative_to(ROOT)
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        elif path.suffix.lower() in {".cff", ".yaml", ".yml"}:
            try:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid YAML {path.relative_to(ROOT)}: {exc}")
        elif path.suffix == ".py":
            try:
                source = path.read_text(encoding="utf-8")
                compile(source, str(path), "exec", dont_inherit=True)
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"python compile failed {path.relative_to(ROOT)}: {exc}")
        elif path.suffix == ".sh":
            proc = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True, check=False
            )
            if proc.returncode:
                errors.append(
                    f"shell syntax failed {path.relative_to(ROOT)}: {proc.stderr.strip()}"
                )
            if not os.access(path, os.X_OK) and rel.parts[:2] != (
                "internal",
                "vendor",
            ):
                warnings.append(
                    f"shell script is not executable: {path.relative_to(ROOT)}"
                )

    required = [
        "README.md",
        "run.sh",
        "setup.sh",
        "LICENSE",
        "NOTICE",
        "pyproject.toml",
        ".github/CONTRIBUTING.md",
        "docs/MODEL_CARD.md",
        "docs/RELEASE_MANIFEST.md",
        "docs/TRAINING_CORRECTNESS.md",
        "src/cd_lam/__init__.py",
        "tests",
        "tests/fixtures/episodes.jsonl",
        "third_party/dependencies.lock.json",
        "scripts/check_wheel.py",
    ]
    for item in required:
        if not (ROOT / item).exists():
            errors.append(f"missing required release item: {item}")

    lock_path = ROOT / "third_party" / "dependencies.lock.json"
    fetch_path = ROOT / "scripts" / "fetch_optional_deps.sh"
    if lock_path.is_file() and fetch_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        fetch_text = fetch_path.read_text(encoding="utf-8")
        for dependency in lock.get("dependencies", []):
            fetch_mode = dependency.get("fetch_mode")
            if fetch_mode not in {"manual", "source-script"}:
                errors.append(
                    f"dependency lock {dependency.get('name', '<unnamed>')} "
                    "fetch_mode must be 'manual' or 'source-script'"
                )
                continue
            if fetch_mode == "manual":
                continue
            for field in ("repository", "revision"):
                value = dependency.get(field)
                if not isinstance(value, str) or value not in fetch_text:
                    errors.append(
                        f"dependency lock {dependency.get('name', '<unnamed>')} "
                        f"{field} is not pinned by fetch_optional_deps.sh"
                    )
            if not dependency.get("runtime_overlay_bundled"):
                continue
            overlay_relative = dependency.get("runtime_overlay_manifest")
            if not isinstance(overlay_relative, str):
                errors.append("bundled runtime overlay has no manifest path")
                continue
            overlay_manifest = ROOT / overlay_relative
            if not overlay_manifest.is_file():
                errors.append(
                    f"runtime overlay manifest is missing: {overlay_relative}"
                )
                continue
            expected_manifest_hash = dependency.get("runtime_overlay_manifest_sha256")
            if _sha256(overlay_manifest) != expected_manifest_hash:
                errors.append("runtime overlay manifest SHA-256 does not match lock")
                continue
            overlay_payload = json.loads(overlay_manifest.read_text(encoding="utf-8"))
            overlay_rows = overlay_payload.get("overlays")
            expected_count = dependency.get("runtime_overlay_files")
            if (
                not isinstance(overlay_rows, list)
                or len(overlay_rows) != expected_count
            ):
                errors.append("runtime overlay file count does not match lock")
                continue
            if overlay_payload.get("base_commit") != dependency.get("revision"):
                errors.append("runtime overlay base commit does not match lock")
            for row in overlay_rows:
                if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                    errors.append("runtime overlay row is invalid")
                    continue
                overlay_file = overlay_manifest.parent / row["path"]
                if (
                    not overlay_file.is_file()
                    or overlay_file.stat().st_size != row.get("bytes")
                    or _sha256(overlay_file) != row.get("sha256")
                ):
                    errors.append(f"runtime overlay file does not match: {row['path']}")

    ci_path = ROOT / ".github" / "workflows" / "ci.yml"
    if ci_path.is_file():
        ci_text = ci_path.read_text(encoding="utf-8")
        required_ci_commands = (
            "python -m ruff check .",
            "python -m pytest -q",
            "python -m cd_lam smoke",
            "python -m cd_lam data-prepare",
            "python -m build --wheel",
            "python scripts/check_wheel.py",
            "https://download.pytorch.org/whl/cu128",
        )
        for command in required_ci_commands:
            if command not in ci_text:
                errors.append(f"CI is missing required gate: {command}")
    else:
        errors.append("missing required release item: .github/workflows/ci.yml")

    for warning in warnings:
        print(f"WARN {warning}")
    for error in errors:
        print(f"ERROR {error}")
    print(
        f"release_check files={len(files)} text_files={text_files} "
        f"cjk_hits={cjk_hits} errors={len(errors)} warnings={len(warnings)}"
    )
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
