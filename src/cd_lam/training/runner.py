"""Shared fail-closed execution and ordered synthetic pipeline runner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

from ..adapters import load_stage_adapter
from ..bridge import load_bridge_checkpoint
from ..config import PipelineConfig, StageName
from ..plans import StagePlan, build_stage_plan
from .common import (
    StageContext,
    StageExecutionError,
    StageResult,
    file_sha256,
    validate_result,
    write_result_metadata,
)
from .synthetic import (
    run_bridge_training,
    run_stage1,
    run_stage2,
    run_stage3,
)


SyntheticExecutor = Callable[[StageContext], StageResult]
_SYNTHETIC_EXECUTORS: Mapping[StageName, SyntheticExecutor] = {
    StageName.STAGE1: run_stage1,
    StageName.STAGE2: run_stage2,
    StageName.BRIDGE: run_bridge_training,
    StageName.STAGE3: run_stage3,
}


def _configured_upstream_artifacts(
    config: PipelineConfig, stage: StageName
) -> dict[str, Path]:
    """Resolve parent model roles already required by an external plan."""

    candidates: Mapping[StageName, Mapping[str, Path | None]] = {
        StageName.STAGE1: {"lam_init": config.paths.lam_init},
        StageName.STAGE2: {
            "base_acwm": config.paths.base_acwm,
            "stage1_lam": config.paths.stage1_lam,
        },
        StageName.BRIDGE: {"stage1_lam": config.paths.stage1_lam},
        StageName.STAGE3: {
            "bridge_bundle": config.paths.bridge_bundle,
            "stage2_acwm": config.paths.stage2_acwm,
        },
    }
    return {
        role: path
        for role, path in candidates[stage].items()
        if path is not None
    }


def execute_stage(
    config: PipelineConfig,
    plan: StagePlan,
    *,
    upstream_artifacts: Mapping[str, Path] | None = None,
) -> StageResult:
    """Execute a ready plan through the synthetic or external adapter boundary."""

    if plan.config_digest != config.digest:
        raise StageExecutionError("plan/config digest mismatch")
    if not plan.ready:
        raise StageExecutionError(
            f"{plan.stage.value} plan is blocked: " + "; ".join(plan.blockers)
        )
    selected_upstreams = upstream_artifacts
    if selected_upstreams is None and plan.mode == "external":
        selected_upstreams = _configured_upstream_artifacts(config, plan.stage)
    context = StageContext(
        config=config,
        plan=plan,
        upstream_artifacts={} if selected_upstreams is None else selected_upstreams,
    )
    if plan.mode == "synthetic":
        result = _SYNTHETIC_EXECUTORS[plan.stage](context)
    else:
        if not plan.adapter_specification:
            raise StageExecutionError("external plan has no adapter specification")
        adapter = load_stage_adapter(plan.adapter_specification, plan.stage)
        effective_plan = replace(plan, adapter_identity=adapter.identity)
        context = replace(context, plan=effective_plan)
        if plan.stage == StageName.STAGE3:
            bridge_path = config.paths.bridge_bundle
            if bridge_path is None:
                raise StageExecutionError("Stage3 bridge bundle is not configured")
            contract = load_bridge_checkpoint(bridge_path)
            actual_transform = contract.metadata.get("action_transform_id")
            actual_stride = contract.metadata.get("source_stride")
            if actual_transform != plan.action_transform_id:
                raise StageExecutionError(
                    "bridge checkpoint action_transform_id mismatch: "
                    f"{actual_transform!r} != {plan.action_transform_id!r}"
                )
            if actual_stride != plan.source_stride:
                raise StageExecutionError(
                    "bridge checkpoint source_stride mismatch: "
                    f"{actual_stride!r} != {plan.source_stride!r}"
                )
        adapter.validate(context)
        result = adapter.run(context)
    validate_result(result, context)
    write_result_metadata(result)
    return result


def run_synthetic_pipeline(
    config: PipelineConfig,
    *,
    output_root: Path,
    target_steps: int = 2,
    seed: int | None = None,
) -> tuple[list[StageResult], Path]:
    """Run Stage1 -> Stage2/bridge -> Stage3 and write an auditable summary."""

    output_root = output_root.expanduser().resolve()
    runtime_config = config.with_paths(output_root=output_root)
    stage_order = (
        StageName.STAGE1,
        StageName.STAGE2,
        StageName.BRIDGE,
        StageName.STAGE3,
    )
    results: list[StageResult] = []
    artifacts: dict[str, Path] = {}

    stage1_plan = build_stage_plan(
        runtime_config,
        StageName.STAGE1,
        synthetic=True,
        target_steps=target_steps,
        seed=seed,
    )
    stage1_result = execute_stage(runtime_config, stage1_plan)
    results.append(stage1_result)
    artifacts["stage1_lam"] = stage1_result.checkpoint

    stage2_plan = build_stage_plan(
        runtime_config,
        StageName.STAGE2,
        synthetic=True,
        target_steps=target_steps,
        seed=seed,
    )
    stage2_result = execute_stage(
        runtime_config,
        stage2_plan,
        upstream_artifacts={"stage1_lam": artifacts["stage1_lam"]},
    )
    results.append(stage2_result)
    artifacts["stage2_acwm"] = stage2_result.checkpoint

    bridge_plan = build_stage_plan(
        runtime_config,
        StageName.BRIDGE,
        synthetic=True,
        target_steps=target_steps,
        seed=seed,
    )
    bridge_result = execute_stage(
        runtime_config,
        bridge_plan,
        upstream_artifacts={"stage1_lam": artifacts["stage1_lam"]},
    )
    results.append(bridge_result)
    artifacts["bridge_bundle"] = bridge_result.checkpoint

    stage3_plan = build_stage_plan(
        runtime_config,
        StageName.STAGE3,
        synthetic=True,
        target_steps=target_steps,
        seed=seed,
    )
    stage3_result = execute_stage(
        runtime_config,
        stage3_plan,
        upstream_artifacts={
            "stage2_acwm": artifacts["stage2_acwm"],
            "bridge_bundle": artifacts["bridge_bundle"],
        },
    )
    results.append(stage3_result)
    artifacts["stage3_acwm"] = stage3_result.checkpoint

    summary = {
        "artifacts": {
            key: {"path": str(path), "sha256": file_sha256(path)}
            for key, path in sorted(artifacts.items())
        },
        "config_digest": config.digest,
        "seed": config.runtime.seed if seed is None else seed,
        "stage_order": [stage.value for stage in stage_order],
        "stages": [result.to_dict() for result in results],
        "status": "pass",
        "steps": target_steps,
    }
    summary_path = output_root / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results, summary_path


__all__ = ["execute_stage", "run_synthetic_pipeline"]
