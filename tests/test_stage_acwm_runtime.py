from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "internal" / "tools" / "stage_acwm_runtime.py"
SPEC = importlib.util.spec_from_file_location("stage_acwm_runtime", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)


def test_runtime_overlay_manifest_is_safe_and_complete() -> None:
    payload = STAGE.load_manifest(STAGE.DEFAULT_MANIFEST)
    rows = payload["overlays"]
    paths = [STAGE.safe_relative(row["path"], "overlay.path") for row in rows]

    assert payload["publication_status"] == "bundled"
    assert payload["base_commit"] == "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff"
    assert len(paths) == 61
    assert payload["runtime_tree"]["algorithm"] == STAGE.RUNTIME_TREE_ALGORITHM
    assert payload["runtime_tree"]["files"] > len(paths)
    assert len(payload["runtime_tree"]["sha256"]) == 64
    assert len(paths) == len(set(paths))
    assert all(len(row["sha256"]) == 64 and row["bytes"] >= 0 for row in rows)
    assert all(row["operation"] in {"added", "modified"} for row in rows)
    assert all(
        row["operation"] == "added" or len(row["base_sha256"]) == 64 for row in rows
    )
    path_strings = {path.as_posix() for path in paths}
    assert {
        "cosmos_predict2/experiments/base/action.py",
        "lamwm_pipline/tools/train_wm_compat_real.py",
        "lamwm_pipline/tools/z_usage_monitor.py",
        "New LAM/Post Train/train_gbridge_z_posttrain.py",
        "New LAM/iterations/_dist/train_a22z_frozen.py",
        "training_scope/LAM/V7/tools/_lam_alias_shim.py",
        "training_scope/LAM/V7/tools/_lam_v7_ctx.py",
        "training_scope/LAM/V7/tools/probe_ee_vs_joint.py",
        "training_scope/LAM/V7/tools/probe_idm_upper_bound.py",
    } <= path_strings


@pytest.mark.parametrize("value", ["/absolute.py", "../escape.py", "a\\b.py", ""])
def test_runtime_overlay_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(STAGE.RuntimeSourceError, match="path"):
        STAGE.safe_relative(value, "test path")


def test_runtime_overlay_verification_detects_tampering(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    source = overlay / "module.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    row = {
        "path": "module.py",
        "bytes": source.stat().st_size,
        "sha256": STAGE.sha256(source),
        "operation": "added",
    }

    observed, relative = STAGE._verify_overlay(overlay, row)
    assert observed == source
    assert relative == Path("module.py")

    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(STAGE.RuntimeSourceError, match="does not match"):
        STAGE._verify_overlay(overlay, row)


def test_runtime_manifest_rejects_bad_commit(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_id": "cdlam-acwm-runtime",
                "publication_status": "bundled",
                "base_repository": "https://github.com/NVIDIA/DreamDojo.git",
                "base_commit": "short",
                "overlays": [{"path": "module.py"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(STAGE.RuntimeSourceError, match="base_commit"):
        STAGE.load_manifest(path)


def test_runtime_stager_applies_added_and_modified_files(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    subprocess.run(["git", "init", "-q", str(base)], check=True)
    subprocess.run(["git", "-C", str(base), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(base), "config", "user.email", "test@example.com"],
        check=True,
    )
    original = base / "modified.py"
    original.write_text("value = 1\n", encoding="utf-8")
    stable = base / "stable.py"
    stable.write_text("stable = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(base), "add", "modified.py", "stable.py"], check=True
    )
    subprocess.run(["git", "-C", str(base), "commit", "-qm", "base"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(base), "rev-parse", "HEAD"], text=True
    ).strip()

    overlay = tmp_path / "overlay"
    overlay.mkdir()
    modified = overlay / "modified.py"
    modified.write_text("value = 2\n", encoding="utf-8")
    added = overlay / "added.py"
    added.write_text("ready = True\n", encoding="utf-8")
    expected = tmp_path / "expected"
    expected.mkdir()
    (expected / "modified.py").write_text("value = 2\n", encoding="utf-8")
    (expected / "stable.py").write_text("stable = True\n", encoding="utf-8")
    (expected / "added.py").write_text("ready = True\n", encoding="utf-8")
    manifest = overlay / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_id": "cdlam-acwm-runtime",
                "publication_status": "bundled",
                "base_repository": "https://github.com/NVIDIA/DreamDojo.git",
                "base_commit": commit,
                "runtime_tree": STAGE.runtime_tree_summary(expected),
                "required_runtime_paths": ["modified.py", "added.py"],
                "overlays": [
                    {
                        "path": "modified.py",
                        "bytes": modified.stat().st_size,
                        "sha256": STAGE.sha256(modified),
                        "operation": "modified",
                        "base_sha256": STAGE.sha256(original),
                    },
                    {
                        "path": "added.py",
                        "bytes": added.stat().st_size,
                        "sha256": STAGE.sha256(added),
                        "operation": "added",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "runtime"

    provenance = STAGE.stage_runtime(base, overlay, output, manifest)

    assert (output / "modified.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (output / "added.py").read_text(encoding="utf-8") == "ready = True\n"
    assert provenance["overlay_files"] == 2
    assert STAGE.verify_runtime(output, manifest) == provenance

    (output / "stable.py").write_text("stable = False\n", encoding="utf-8")
    with pytest.raises(STAGE.RuntimeSourceError, match="runtime tree drifted"):
        STAGE.verify_runtime(output, manifest)
    (output / "stable.py").write_text("stable = True\n", encoding="utf-8")

    (output / "added.py").write_text("tampered = True\n", encoding="utf-8")
    with pytest.raises(STAGE.RuntimeSourceError, match="drifted"):
        STAGE.verify_runtime(output, manifest)
