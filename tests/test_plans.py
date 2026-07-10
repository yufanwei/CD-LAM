from __future__ import annotations

from dataclasses import replace

from cd_lam.config import PipelineConfig, StageName
from cd_lam.plans import build_stage_plan


def test_external_plan_fails_closed_without_assets_or_adapter(tmp_path) -> None:
    config = PipelineConfig.synthetic(tmp_path / "outputs")
    plan = build_stage_plan(config, StageName.STAGE2)
    assert not plan.ready
    joined = "\n".join(plan.blockers)
    assert "no external adapter" in joined
    assert "unlabeled_manifest" in joined
    assert "base_acwm" in joined
    assert plan.to_json() == plan.to_json()


def test_synthetic_plan_is_cpu_only_and_ready(tmp_path) -> None:
    config = PipelineConfig.synthetic(tmp_path / "outputs", seed=9)
    plan = build_stage_plan(
        config, StageName.BRIDGE, synthetic=True, target_steps=3
    )
    assert plan.ready
    assert plan.device == "cpu"
    assert plan.seed == 9
    assert plan.target_steps == 3
    assert plan.adapter_identity == "cd_lam.synthetic_cpu"


def test_bridge_and_stage3_transform_mismatch_blocks_real_execution(tmp_path) -> None:
    config = PipelineConfig.synthetic(tmp_path / "outputs")
    mismatched = replace(
        config,
        bridge=replace(
            config.bridge,
            action_transform_id="raw_delta_stride4",
            source_stride=4,
            adapter="fake:adapter",
        ),
        stage3=replace(
            config.stage3,
            action_transform_id="minmax_absolute_then_delta_stride4",
            source_stride=4,
            adapter="fake:adapter",
            working_directory=tmp_path,
        ),
    )
    plan = build_stage_plan(mismatched, StageName.STAGE3)
    assert not plan.ready
    assert "action_transform_id mismatch" in "\n".join(plan.blockers)


def test_stage3_real_plan_requires_explicit_working_directory(tmp_path) -> None:
    config = PipelineConfig.synthetic(tmp_path / "outputs")
    configured = replace(
        config,
        stage3=replace(config.stage3, adapter="fake:adapter", working_directory=None),
    )
    plan = build_stage_plan(configured, StageName.STAGE3)
    assert "working_directory is required" in "\n".join(plan.blockers)
