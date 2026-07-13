#!/usr/bin/env python3
"""Validate the unified CD-LAM GPU environment without loading models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "configs/model_runtime.lock.json"
DEFAULT_ENVIRONMENT = ROOT / ".venv"
DEFAULT_ACWM_ROOT = ROOT / ".deps/acwm-runtime"
DEFAULT_OVERLAY_MANIFEST = ROOT / "third_party/acwm_overlay/manifest.json"
DEFAULT_SOURCE_VERIFIER = ROOT / "internal/tools/stage_acwm_runtime.py"


class DoctorError(RuntimeError):
    """The model-runtime contract or one of its inputs is invalid."""


def sha256_file(path: Path) -> str:
    """Hash one regular file without following a caller-supplied directory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path) -> dict[str, object]:
    """Load and minimally validate the checked-in model-runtime lock."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DoctorError(f"cannot read model-runtime lock {path}: {exc}") from exc
    if document.get("schema_version") != 1:
        raise DoctorError("model-runtime lock schema_version must be 1")
    for key in (
        "installer",
        "source",
        "platform",
        "critical_distributions",
        "required_modules",
        "allowed_metadata_conflicts",
        "editable_sources",
        "extra_sources",
    ):
        if key not in document:
            raise DoctorError(f"model-runtime lock is missing {key!r}")
    return document


def _json_file(path: Path, label: str, errors: list[str]) -> dict[str, object] | None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label} {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a JSON object: {path}")
        return None
    return value


def source_errors(
    lock: dict[str, object],
    acwm_root: Path,
    overlay_manifest_path: Path = DEFAULT_OVERLAY_MANIFEST,
    *,
    verify_runtime_tree: bool = False,
    source_verifier: Path = DEFAULT_SOURCE_VERIFIER,
) -> list[str]:
    """Validate the staged upstream source and its complete transitive lock."""

    errors: list[str] = []
    source = lock["source"]
    assert isinstance(source, dict)
    revision = source.get("revision")
    repository = source.get("repository")

    uv_lock = acwm_root / "uv.lock"
    if not uv_lock.is_file():
        errors.append(f"missing upstream uv.lock: {uv_lock}")
    else:
        observed = sha256_file(uv_lock)
        expected = source.get("uv_lock_sha256")
        if observed != expected:
            errors.append(
                f"upstream uv.lock SHA-256 mismatch: expected {expected}, got {observed}"
            )

    provenance = _json_file(
        acwm_root / ".cdlam-runtime-source.json",
        "staged-runtime provenance",
        errors,
    )
    overlay = _json_file(overlay_manifest_path, "overlay manifest", errors)
    if provenance is not None:
        if provenance.get("base_commit") != revision:
            errors.append("staged runtime base commit does not match the model lock")
        if provenance.get("base_repository") != repository:
            errors.append("staged runtime repository does not match the model lock")
    if overlay is not None:
        if overlay.get("base_commit") != revision:
            errors.append("overlay base commit does not match the model lock")
        if overlay.get("base_repository") != repository:
            errors.append("overlay repository does not match the model lock")
    if provenance is not None and overlay_manifest_path.is_file():
        expected_overlay = sha256_file(overlay_manifest_path)
        if provenance.get("overlay_manifest_sha256") != expected_overlay:
            errors.append(
                "staged runtime was not built from the current overlay manifest"
            )

    for relative in (
        "pyproject.toml",
        "packages/cosmos-oss/pyproject.toml",
        "packages/cosmos-cuda/pyproject.toml",
        "cosmos_predict2/__about__.py",
    ):
        path = acwm_root / relative
        if not path.is_file():
            errors.append(f"missing staged runtime source file: {path}")
    if verify_runtime_tree:
        errors.extend(
            runtime_tree_errors(acwm_root, overlay_manifest_path, source_verifier)
        )
    return errors


def runtime_tree_errors(
    acwm_root: Path,
    overlay_manifest_path: Path = DEFAULT_OVERLAY_MANIFEST,
    source_verifier: Path = DEFAULT_SOURCE_VERIFIER,
) -> list[str]:
    """Run the canonical full-tree verifier, including unmodified base files."""

    if not source_verifier.is_file():
        return [f"missing complete runtime-tree verifier: {source_verifier}"]
    result = subprocess.run(
        [
            sys.executable,
            str(source_verifier),
            "--verify-existing",
            "--output",
            str(acwm_root),
            "--manifest",
            str(overlay_manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode == 0:
        return []
    detail = (
        result.stderr.strip() or result.stdout.strip() or "unknown verifier failure"
    )
    return [f"complete staged runtime verification failed: {detail}"]


def _environment_probe(
    environment: Path,
    distributions: list[str],
    modules: list[str],
    acwm_root: Path,
) -> dict[str, object]:
    python = environment / "bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise DoctorError(f"model Python is not executable: {python}")
    program = r"""
import importlib.metadata as metadata
import contextlib
import importlib
import io
import json
import platform
import sys

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

distributions = json.loads(sys.argv[1])
modules = json.loads(sys.argv[2])
versions = {}
direct_urls = {}
for name in distributions:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    versions[name] = distribution.version
    raw = distribution.read_text("direct_url.json")
    if raw:
        direct_urls[name] = json.loads(raw)

installed = {}
duplicate_distributions = []
metadata_errors = []
for distribution in metadata.distributions():
    raw_name = distribution.metadata.get("Name")
    if not raw_name:
        metadata_errors.append("installed distribution has no Name metadata")
        continue
    name = canonicalize_name(raw_name)
    if name in installed:
        duplicate_distributions.append(name)
        continue
    installed[name] = distribution

marker_environment = default_environment()
marker_environment["extra"] = ""
metadata_conflicts = []
for requiring_name, distribution in sorted(installed.items()):
    for raw_requirement in distribution.requires or []:
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement as exc:
            metadata_errors.append(
                f"{requiring_name} has invalid requirement {raw_requirement!r}: {exc}"
            )
            continue
        if requirement.marker is not None and not requirement.marker.evaluate(
            marker_environment
        ):
            continue
        required_name = canonicalize_name(requirement.name)
        installed_requirement = installed.get(required_name)
        if installed_requirement is None:
            metadata_conflicts.append(
                {
                    "kind": "missing",
                    "requiring_distribution": requiring_name,
                    "requiring_version": distribution.version,
                    "required_distribution": required_name,
                    "specifier": str(requirement.specifier),
                    "installed_version": None,
                }
            )
            continue
        if not requirement.specifier:
            continue
        try:
            satisfies = requirement.specifier.contains(
                Version(installed_requirement.version), prereleases=True
            )
        except InvalidVersion as exc:
            metadata_errors.append(
                f"{required_name} has invalid version {installed_requirement.version!r}: {exc}"
            )
            continue
        if not satisfies:
            metadata_conflicts.append(
                {
                    "kind": "version",
                    "requiring_distribution": requiring_name,
                    "requiring_version": distribution.version,
                    "required_distribution": required_name,
                    "specifier": str(requirement.specifier),
                    "installed_version": installed_requirement.version,
                }
            )

module_imports = {}
for name in modules:
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
            captured_stderr
        ):
            importlib.import_module(name)
    except BaseException as exc:
        module_imports[name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        module_imports[name] = {"ok": True, "error": None}
payload = {
    "python": platform.python_version(),
    "implementation": platform.python_implementation(),
    "system": platform.system(),
    "machine": platform.machine(),
    "libc": platform.libc_ver()[1],
    "versions": versions,
    "direct_urls": direct_urls,
    "module_imports": module_imports,
    "metadata_conflicts": sorted(
        metadata_conflicts,
        key=lambda row: (
            row["requiring_distribution"],
            row["required_distribution"],
            row["specifier"],
        ),
    ),
    "metadata_errors": sorted(metadata_errors),
    "duplicate_distributions": sorted(set(duplicate_distributions)),
}
print(json.dumps(payload, sort_keys=True))
"""
    environment_variables = os.environ.copy()
    environment_variables.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(acwm_root),
                    str(acwm_root / "packages/cosmos-cuda"),
                    str(acwm_root / "packages/cosmos-oss"),
                ]
            ),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    result = subprocess.run(
        [str(python), "-c", program, json.dumps(distributions), json.dumps(modules)],
        check=False,
        capture_output=True,
        text=True,
        env=environment_variables,
    )
    if result.returncode != 0:
        detail = (
            result.stderr.strip() or result.stdout.strip() or "unknown probe failure"
        )
        raise DoctorError(f"model environment probe failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DoctorError("model environment probe did not return JSON") from exc
    if not isinstance(payload, dict):
        raise DoctorError("model environment probe returned an invalid payload")
    return payload


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value)
    return tuple(int(number) for number in numbers)


def _file_url_path(value: object) -> Path | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


_CONFLICT_FIELDS = (
    "kind",
    "requiring_distribution",
    "requiring_version",
    "required_distribution",
    "specifier",
    "installed_version",
)


def _conflict_key(value: object) -> tuple[object, ...] | None:
    if not isinstance(value, dict):
        return None
    if value.get("kind") not in {"missing", "version"}:
        return None
    if any(field not in value for field in _CONFLICT_FIELDS):
        return None
    return tuple(value[field] for field in _CONFLICT_FIELDS)


def metadata_conflict_errors(
    lock: dict[str, object], payload: dict[str, object]
) -> list[str]:
    """Require observed dependency conflicts to equal the explicit lock allowlist."""

    errors: list[str] = []
    expected_rows = lock.get("allowed_metadata_conflicts")
    observed_rows = payload.get("metadata_conflicts")
    if not isinstance(expected_rows, list):
        return ["model-runtime lock has no metadata-conflict allowlist"]
    if not isinstance(observed_rows, list):
        return ["environment probe has no structured metadata-conflict inventory"]

    expected: set[tuple[object, ...]] = set()
    observed: set[tuple[object, ...]] = set()
    for row in expected_rows:
        key = _conflict_key(row)
        if key is None:
            errors.append(f"invalid allowed metadata conflict: {row!r}")
        elif key in expected:
            errors.append(f"duplicate allowed metadata conflict: {row!r}")
        else:
            expected.add(key)
    for row in observed_rows:
        key = _conflict_key(row)
        if key is None:
            errors.append(f"invalid observed metadata conflict: {row!r}")
        elif key in observed:
            errors.append(f"duplicate observed metadata conflict: {row!r}")
        else:
            observed.add(key)

    for key in sorted(observed - expected, key=repr):
        errors.append(
            f"unexpected dependency metadata conflict: {dict(zip(_CONFLICT_FIELDS, key, strict=True))}"
        )
    for key in sorted(expected - observed, key=repr):
        errors.append(
            f"expected dependency metadata conflict is missing: {dict(zip(_CONFLICT_FIELDS, key, strict=True))}"
        )

    metadata_errors = payload.get("metadata_errors")
    if not isinstance(metadata_errors, list):
        errors.append("environment probe has no metadata parse-error inventory")
    else:
        errors.extend(
            f"dependency metadata error: {value}" for value in metadata_errors
        )
    duplicates = payload.get("duplicate_distributions")
    if not isinstance(duplicates, list):
        errors.append("environment probe has no duplicate-distribution inventory")
    else:
        errors.extend(
            f"duplicate installed distribution: {value}" for value in duplicates
        )
    return errors


def probe_errors(
    lock: dict[str, object],
    payload: dict[str, object],
    acwm_root: Path,
) -> list[str]:
    """Compare a target-interpreter probe with the checked-in contract."""

    errors: list[str] = []
    platform_lock = lock["platform"]
    expected_versions = lock["critical_distributions"]
    editable_sources = lock["editable_sources"]
    extra_sources = lock["extra_sources"]
    assert isinstance(platform_lock, dict)
    assert isinstance(expected_versions, dict)
    assert isinstance(editable_sources, dict)
    assert isinstance(extra_sources, dict)

    for key in ("system", "machine", "python_implementation"):
        payload_key = "implementation" if key == "python_implementation" else key
        if payload.get(payload_key) != platform_lock.get(key):
            errors.append(
                f"{payload_key} mismatch: expected {platform_lock.get(key)}, "
                f"got {payload.get(payload_key)}"
            )
    python_version = str(payload.get("python", ""))
    if ".".join(python_version.split(".")[:2]) != platform_lock.get(
        "python_major_minor"
    ):
        errors.append(
            f"Python mismatch: expected {platform_lock.get('python_major_minor')}.x, "
            f"got {python_version}"
        )
    libc = str(payload.get("libc", ""))
    minimum_libc = str(platform_lock.get("glibc_minimum", ""))
    if not libc or _version_tuple(libc) < _version_tuple(minimum_libc):
        errors.append(f"glibc {libc or '<unknown>'} is older than {minimum_libc}")

    versions = payload.get("versions")
    module_imports = payload.get("module_imports")
    direct_urls = payload.get("direct_urls")
    if not isinstance(versions, dict):
        return errors + ["environment probe has no distribution versions"]
    if not isinstance(module_imports, dict):
        return errors + ["environment probe has no module-import inventory"]
    if not isinstance(direct_urls, dict):
        return errors + ["environment probe has no direct-source inventory"]

    for name, expected in expected_versions.items():
        observed = versions.get(name)
        if observed != expected:
            errors.append(
                f"distribution {name} mismatch: expected {expected}, got {observed}"
            )
    for name in lock["required_modules"]:
        result = module_imports.get(name)
        if not isinstance(result, dict) or result.get("ok") is not True:
            detail = result.get("error") if isinstance(result, dict) else "not probed"
            errors.append(f"required module import failed: {name}: {detail}")

    errors.extend(metadata_conflict_errors(lock, payload))

    for name, relative in editable_sources.items():
        direct = direct_urls.get(name)
        url = direct.get("url") if isinstance(direct, dict) else None
        observed = _file_url_path(url)
        expected = (acwm_root / str(relative)).resolve()
        if observed != expected:
            errors.append(
                f"distribution {name} is not installed from staged source: "
                f"expected {expected}, got {observed}"
            )

    pytorch3d = extra_sources.get("pytorch3d")
    direct = direct_urls.get("pytorch3d")
    if isinstance(pytorch3d, dict) and isinstance(direct, dict):
        vcs = direct.get("vcs_info")
        observed = vcs.get("commit_id") if isinstance(vcs, dict) else None
        if observed != pytorch3d.get("revision"):
            errors.append("pytorch3d source revision does not match the model lock")
    else:
        errors.append("pytorch3d direct-source metadata is missing")
    return errors


def environment_errors(
    lock: dict[str, object], environment: Path, acwm_root: Path
) -> tuple[list[str], dict[str, object] | None]:
    """Validate isolation, executable availability, packages, and import specs."""

    errors: list[str] = []
    configuration = environment / "pyvenv.cfg"
    if not configuration.is_file():
        errors.append(f"missing virtual-environment metadata: {configuration}")
    else:
        text = configuration.read_text(encoding="utf-8")
        if not re.search(
            r"^include-system-site-packages\s*=\s*false\s*$", text, re.MULTILINE
        ):
            errors.append("model environment must not include system site packages")
    torchrun = environment / "bin/torchrun"
    if not torchrun.is_file() or not os.access(torchrun, os.X_OK):
        errors.append(f"torchrun is not executable: {torchrun}")
    try:
        payload = _environment_probe(
            environment,
            list(lock["critical_distributions"]),
            list(lock["required_modules"]),
            acwm_root,
        )
    except DoctorError as exc:
        errors.append(str(exc))
        return errors, None
    errors.extend(probe_errors(lock, payload, acwm_root))
    return errors, payload


def driver_errors(
    lock: dict[str, object], environment: Path, gpu: int
) -> tuple[list[str], dict[str, object] | None]:
    """Check driver visibility and CUDA compatibility without launching a kernel."""

    errors: list[str] = []
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=driver_version,name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        detail = query.stderr.strip() or "nvidia-smi failed"
        return [detail], None
    first = query.stdout.strip().splitlines()[0]
    driver, _, name = first.partition(",")
    driver = driver.strip()
    name = name.strip()
    platform_lock = lock["platform"]
    assert isinstance(platform_lock, dict)
    minimum = str(platform_lock["nvidia_driver_minimum"])
    if _version_tuple(driver) < _version_tuple(minimum):
        errors.append(f"NVIDIA driver {driver} is older than required {minimum}")

    program = r"""
import json
import torch

payload = {
    "torch": torch.__version__,
    "compiled_cuda": torch.version.cuda,
    "available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if payload["available"] and payload["device_count"]:
    payload["device"] = torch.cuda.get_device_name(0)
print(json.dumps(payload, sort_keys=True))
"""
    environment_variables = os.environ.copy()
    environment_variables.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    result = subprocess.run(
        [str(environment / "bin/python"), "-c", program],
        check=False,
        capture_output=True,
        text=True,
        env=environment_variables,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "PyTorch CUDA probe failed"
        return errors + [detail], None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return errors + ["PyTorch CUDA probe did not return JSON"], None
    if payload.get("compiled_cuda") != platform_lock.get("cuda_wheel_line"):
        errors.append(
            f"PyTorch CUDA mismatch: expected {platform_lock.get('cuda_wheel_line')}, "
            f"got {payload.get('compiled_cuda')}"
        )
    if payload.get("available") is not True or payload.get("device_count", 0) < 1:
        errors.append("CUDA is not available to the model interpreter")
    payload.update({"driver": driver, "nvidia_smi_device": name})
    return errors, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--acwm-root", type=Path, default=DEFAULT_ACWM_ROOT)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--check-driver", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be a non-negative integer")

    try:
        lock = load_lock(args.lock.resolve())
    except DoctorError as exc:
        parser.exit(2, f"model_runtime_doctor: {exc}\n")
    acwm_root = args.acwm_root.expanduser().resolve()
    environment = args.environment.expanduser().resolve()
    errors = source_errors(lock, acwm_root, verify_runtime_tree=True)
    details: dict[str, object] = {
        "profile": lock.get("profile"),
        "acwm_root": str(acwm_root),
    }
    if not args.source_only:
        environment_failures, probe = environment_errors(lock, environment, acwm_root)
        errors.extend(environment_failures)
        details["environment"] = str(environment)
        if probe is not None:
            details["python"] = probe.get("python")
            details["critical_distributions"] = probe.get("versions")
            details["dependency_metadata_conflicts"] = probe.get("metadata_conflicts")
            details["required_module_imports"] = probe.get("module_imports")
    if args.check_driver:
        if args.source_only:
            errors.append("--check-driver cannot be combined with --source-only")
        else:
            cuda_failures, cuda = driver_errors(lock, environment, args.gpu)
            errors.extend(cuda_failures)
            if cuda is not None:
                details["cuda"] = cuda

    status = "pass" if not errors else "fail"
    if args.json:
        print(
            json.dumps(
                {"status": status, "errors": errors, "details": details},
                indent=2,
                sort_keys=True,
            )
        )
    elif errors:
        print("model_runtime_doctor: FAIL", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
    else:
        print(f"model_runtime_doctor: PASS ({lock.get('profile')})")
        print(f"  staged source: {acwm_root}")
        if not args.source_only:
            print(f"  environment:   {environment}")
            print(
                "  metadata:      "
                f"{len(lock['allowed_metadata_conflicts'])} locked conflicts, "
                "no others"
            )
            print(f"  imports:       {len(lock['required_modules'])} required modules")
        if args.check_driver and "cuda" in details:
            cuda = details["cuda"]
            assert isinstance(cuda, dict)
            print(
                f"  CUDA:         {cuda.get('compiled_cuda')} on "
                f"{cuda.get('device')} (driver {cuda.get('driver')})"
            )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
