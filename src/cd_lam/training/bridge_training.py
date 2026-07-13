"""Action-to-latent bridge-training planning and execution."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from ..config import PipelineConfig, StageName
from ..plans import StagePlan, build_stage_plan
from .common import StageResult
from .runner import execute_stage


def plan_bridge_training(
    config: PipelineConfig,
    *,
    synthetic: bool = False,
    target_steps: Optional[int] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    adapter: Optional[str] = None,
    resume_from: Optional[Path] = None,
) -> StagePlan:
    return build_stage_plan(
        config,
        StageName.BRIDGE,
        synthetic=synthetic,
        target_steps=target_steps,
        device=device,
        seed=seed,
        adapter=adapter,
        resume_from=resume_from,
    )


def execute_bridge_training(
    config: PipelineConfig,
    plan: StagePlan,
    *,
    upstream_artifacts: Mapping[str, Path] | None = None,
) -> StageResult:
    return execute_stage(config, plan, upstream_artifacts=upstream_artifacts)


__all__ = ["execute_bridge_training", "plan_bridge_training"]
