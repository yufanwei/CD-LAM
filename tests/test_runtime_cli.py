from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from cdlam_runtime.action_contract import (
    EXPECTED_LAYOUT,
    ActionContractError,
    load_stage3_action_contract,
    sha256_file,
)
from cdlam_runtime.bind_bridge import bind_bridge_contract
from cdlam_runtime.config import RuntimeConfig
from cdlam_runtime.runtime import (
    ENTRY_ROOT,
    _external_files,
    runtime_environment,
    write_registry,
)
import cdlam_runtime.runtime as runtime_module
from cdlam_runtime.cli import _parser

ROOT = Path(__file__).resolve().parents[1]
TEST_ACWM_ROOT = os.environ.get("CDLAM_TEST_ACWM_ROOT")


def test_training_commands_run_without_gpu_acknowledgement_flag() -> None:
    for command in ("stage1", "bridge", "stage2", "stage3", "pipeline"):
        args = _parser().parse_args([command])
        assert args.command == command
        assert args.dry_run is False
        assert not hasattr(args, "allow_gpu")

    with pytest.raises(SystemExit):
        _parser().parse_args(["pipeline", "--allow-gpu"])


def _profile(tmp_path: Path, acwm: Path) -> Path:
    document = json.loads((ROOT / "configs" / "runtime.example.json").read_text())
    document["workspace"] = str(tmp_path)
    paths = document["paths"]
    paths.update(
        {
            "python": "/usr/bin/python3",
            "torchrun": "/usr/bin/true",
            "acwm_root": str(acwm),
            "hf_home": str(tmp_path / "hf"),
            "base_lam_checkpoint": str(tmp_path / "base-lam.pt"),
            "base_world_checkpoint": str(tmp_path / "world-model"),
            "action_contract": str(ROOT / "configs" / "action_contract.json"),
            "stage1_checkpoint": str(tmp_path / "lam.pt"),
            "output_root": str(tmp_path / "outputs"),
        }
    )
    profile = tmp_path / "runtime.json"
    profile.write_text(json.dumps(document), encoding="utf-8")
    return profile


def test_profile_resolves_relative_paths_from_workspace(tmp_path: Path) -> None:
    profile = _profile(tmp_path, tmp_path / "acwm")
    config = RuntimeConfig.load(profile)
    assert config.workspace == tmp_path
    assert config.path("stage1_train_index") == (
        tmp_path / "data/prepared/stage1/train.parquet"
    )


def test_profile_preserves_unified_environment_python_symlink(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.write_text("fixture", encoding="utf-8")
    environment_python = tmp_path / ".venv" / "bin" / "python"
    environment_python.parent.mkdir(parents=True)
    environment_python.symlink_to(target)
    profile = _profile(tmp_path, tmp_path / "acwm")
    document = json.loads(profile.read_text(encoding="utf-8"))
    document["paths"]["python"] = ".venv/bin/python"
    profile.write_text(json.dumps(document), encoding="utf-8")

    config = RuntimeConfig.load(profile)

    assert config.path("python") == environment_python
    assert config.path("python").is_symlink()


def test_runtime_environment_declares_cdlam_namespace(tmp_path: Path) -> None:
    profile = _profile(tmp_path, tmp_path / "acwm")
    config = RuntimeConfig.load(profile)
    env = runtime_environment(config, tmp_path / "lam.pt")
    assert env["CDLAM_ACWM_ROOT"] == str(tmp_path / "acwm")


def test_world_model_doctor_imports_transitive_data_runtime() -> None:
    stage2 = runtime_module._runtime_import_modules(("stage2",))
    stage3 = runtime_module._runtime_import_modules(("stage3",))
    assert "groot_dreams.dataloader" in stage2
    assert "groot_dreams.dataloader" in stage3
    assert "groot_dreams.dataloader" not in runtime_module._runtime_import_modules(
        ("stage1",)
    )


def test_generated_registry_has_public_ids(tmp_path: Path) -> None:
    profile = _profile(tmp_path, tmp_path / "acwm")
    config = RuntimeConfig.load(profile)
    registry = write_registry(config, tmp_path / "lam.pt")
    text = registry.read_text(encoding="utf-8")
    assert "base_lam:" in text
    assert "cdlam_lam:" in text


def test_full_doctor_does_not_require_pipeline_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path, tmp_path / "acwm")
    document = json.loads(profile.read_text())
    paths = document["paths"]
    paths["python"] = "/bin/true"
    paths["torchrun"] = "/bin/true"
    for key in (
        "base_lam_checkpoint",
        "stage1_recipe",
        "stage1_train_index",
        "stage1_eval_index",
        "bridge_cache",
        "stage2_manifest",
        "stage3_dataset_list",
        "action_contract",
    ):
        path = tmp_path / f"{key}.data"
        path.write_text("fixture", encoding="utf-8")
        paths[key] = str(path)
    for key in ("stage1_checkpoint", "bridge_checkpoint", "stage2_checkpoint"):
        paths[key] = str(tmp_path / f"missing-{key}.pt")
    profile.write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "acwm").mkdir()
    config = RuntimeConfig.load(profile)

    monkeypatch.setattr(runtime_module, "_external_files", lambda *_: [])
    monkeypatch.setattr(runtime_module, "_base_world_errors", lambda *_: [])
    monkeypatch.setattr(runtime_module, "_hf_errors", lambda *_: [])
    monkeypatch.setattr(
        runtime_module, "_validate_public_experiment_aliases", lambda *_: None
    )
    monkeypatch.setattr(runtime_module, "validate_eval_source", lambda *_: None)
    monkeypatch.setattr(runtime_module, "validate_stage2_source", lambda *_: None)
    monkeypatch.setattr(runtime_module, "_validate_contract_assets", lambda *_: None)
    assert runtime_module.doctor(config, "all", imports=False) == []


def test_new_bridge_requires_atomic_contract_binding(tmp_path: Path) -> None:
    external = tmp_path / "acwm"
    metadata_root = external / "shared_meta"
    metadata_root.mkdir(parents=True)
    stats_path = metadata_root / "AgiBot_stats.json"
    modality_path = metadata_root / "AgiBot_modality.json"
    stats_path.write_text(
        json.dumps(
            {
                "action": {
                    "mean": [0.0] * 22,
                    "std": [1.0] * 22,
                    "min": [-1.0] * 22,
                    "max": [1.0] * 22,
                }
            }
        ),
        encoding="utf-8",
    )
    modality_path.write_text(
        json.dumps(
            {
                "action": {
                    name: {
                        "original_key": "action",
                        "start": start,
                        "end": end,
                        "absolute": True,
                    }
                    for name, start, end in EXPECTED_LAYOUT
                }
            }
        ),
        encoding="utf-8",
    )
    contract = json.loads((ROOT / "configs" / "action_contract.json").read_text())
    contract["stats_sha256"] = sha256_file(stats_path)
    contract["modality_sha256"] = sha256_file(modality_path)
    contract["legacy_bridge_sha256_by_scale_family"] = {"100h": "0" * 64}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    bridge_path = tmp_path / "bridge.pt"
    torch.save(
        {
            "g_state": {
                "0.weight": torch.zeros(256, 22),
                "4.weight": torch.zeros(32, 256),
            },
            "action_mean": torch.zeros(22),
            "action_std": torch.ones(22),
            "zm": torch.zeros(32),
            "zsd": torch.ones(32),
            "latent_dim": 32,
        },
        bridge_path,
    )
    stage1 = tmp_path / "lam.pt"
    stage1.write_bytes(b"stage1-checkpoint")

    with pytest.raises(ActionContractError):
        load_stage3_action_contract(contract_path, external, bridge_path, "100h")
    metadata = bind_bridge_contract(
        bridge_path,
        contract_path,
        external,
        stage1,
        lineage="100h",
    )
    validated = load_stage3_action_contract(
        contract_path,
        external,
        bridge_path,
        "100h",
    )
    assert metadata["contract_id"] == validated.contract_id
    assert metadata["stage1_checkpoint_sha256"] == sha256_file(stage1)


@pytest.mark.skipif(
    not TEST_ACWM_ROOT,
    reason="set CDLAM_TEST_ACWM_ROOT for the staged-overlay contract check",
)
def test_staged_overlay_contains_runtime_closure(tmp_path: Path) -> None:
    assert TEST_ACWM_ROOT is not None
    root = Path(TEST_ACWM_ROOT).resolve()
    profile = _profile(tmp_path, root)
    config = RuntimeConfig.load(profile)
    missing = [
        (label, path)
        for stage in ("stage1", "bridge", "stage2", "stage3")
        for label, path in _external_files(config, stage)
        if not path.is_file()
    ]
    assert not missing
    assert all(
        (ENTRY_ROOT / name).is_file()
        for name in ("stage1.py", "stage2.py", "stage3.py")
    )
