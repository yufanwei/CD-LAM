#!/usr/bin/env python3
"""Static release checks that do not need model weights or a GPU."""

from __future__ import annotations

import argparse
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
    "__pycache__",
    "build",
    "dist",
    "outputs",
}
TEXT_SUFFIXES = {
    ".cff", ".cfg", ".env", ".ini", ".json", ".md", ".py", ".sh",
    ".patch", ".toml", ".txt", ".yaml", ".yml",
}
TEXT_FILENAMES = {".gitattributes", ".gitignore", "LICENSE", "Makefile", "NOTICE"}
CJK_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
FORBIDDEN = {
    "private workspace path": re.compile(r"/(?:workspace|home)/yufan|/tmp/data"),
    "dated temporary tool": re.compile(r"tmp_remote_eval_tools_\\d+"),
    "private runtime namespace": re.compile(r"DREAMDOJO_ROOT|DREAMDOJO_REPO"),
    "credential material": re.compile(
        r"(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)"
    ),
}


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
    if top_level.returncode or Path(top_level.stdout.strip()).resolve() != ROOT.resolve():
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
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    text_files = 0
    cjk_hits = 0
    files = tracked_files()

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
        if rel == Path("tools/release_check.py"):
            continue
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                errors.append(f"{label}: {rel}")

    for path in files:
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
            if not os.access(path, os.X_OK):
                warnings.append(f"shell script is not executable: {path.relative_to(ROOT)}")

    required = [
        "README.md", "LICENSE", "NOTICE", "CITATION.cff", "pyproject.toml",
        "docs/TRAINING_CORRECTNESS.md", "src/cd_lam/__init__.py", "tests",
        "test_data/episodes.jsonl", "third_party/dependencies.lock.json",
        "tools/check_wheel.py",
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

    ci_path = ROOT / ".github" / "workflows" / "ci.yml"
    if ci_path.is_file():
        ci_text = ci_path.read_text(encoding="utf-8")
        required_ci_commands = (
            "python -m ruff check .",
            "python -m pytest -q",
            "python -m cd_lam smoke",
            "python -m cd_lam data-prepare",
            "python -m cd_lam train-smoke",
            "python -m build --wheel",
            "python tools/check_wheel.py",
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
