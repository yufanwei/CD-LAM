from __future__ import annotations

import json
from pathlib import Path

import torch

from cd_lam.config import PipelineConfig, StageName
from cd_lam.plans import build_stage_plan
from cd_lam.training.runner import execute_stage, run_synthetic_pipeline


def _metadata(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["metadata"]


def test_ordered_synthetic_pipeline_trains_and_chains_real_artifacts(tmp_path) -> None:
    output_root = tmp_path / "smoke"
    config = PipelineConfig.synthetic(output_root, seed=5)
    results, summary_path = run_synthetic_pipeline(
        config, output_root=output_root, target_steps=1
    )

    assert [result.stage for result in results] == [
        StageName.STAGE1,
        StageName.STAGE2,
        StageName.BRIDGE,
        StageName.STAGE3,
    ]
    assert all(result.checkpoint.is_file() for result in results)
    assert all(result.final_loss <= result.initial_loss * 1.25 for result in results)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["stage_order"] == [
        "stage1",
        "stage2",
        "bridge-train",
        "stage3",
    ]

    stage2_metadata = _metadata(results[1].checkpoint)
    bridge_metadata = _metadata(results[2].checkpoint)
    stage3_metadata = _metadata(results[3].checkpoint)
    assert stage2_metadata["upstream_hashes"]["stage1_lam"]
    assert bridge_metadata["upstream_hashes"]["stage1_lam"]
    assert set(stage3_metadata["upstream_hashes"]) == {
        "bridge_bundle",
        "stage2_acwm",
    }
    for metadata in (stage2_metadata, bridge_metadata, stage3_metadata):
        assert metadata["model_scale"] == "synthetic"
        assert metadata["steps"] == 1
        assert metadata["adapter_identity"] == "cd_lam.synthetic_cpu"


def test_synthetic_checkpoint_resume_records_total_steps(tmp_path) -> None:
    first_config = PipelineConfig.synthetic(tmp_path / "first", seed=3)
    first_plan = build_stage_plan(
        first_config, StageName.STAGE1, synthetic=True, target_steps=1
    )
    first = execute_stage(first_config, first_plan)

    resumed_config = first_config.with_paths(output_root=tmp_path / "resumed")
    resumed_plan = build_stage_plan(
        resumed_config,
        StageName.STAGE1,
        synthetic=True,
        target_steps=3,
        resume_from=first.checkpoint,
    )
    resumed = execute_stage(resumed_config, resumed_plan)
    checkpoint = torch.load(resumed.checkpoint, map_location="cpu", weights_only=False)

    assert resumed.start_step == 1
    assert resumed.steps == 3
    assert checkpoint["metadata"]["steps"] == 3
    assert (
        checkpoint["metadata"]["protocol_steps"] == first_config.stage1.protocol_steps
    )
    assert checkpoint["optimizer_state"]["state"]
