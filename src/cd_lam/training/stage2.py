"""Stage-2 planning and execution behind the external ACWM adapter boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from ..config import PipelineConfig, StageName
from ..plans import StagePlan, build_stage_plan
from .common import StageResult
from .runner import execute_stage


def plan_stage2(
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
        StageName.STAGE2,
        synthetic=synthetic,
        target_steps=target_steps,
        device=device,
        seed=seed,
        adapter=adapter,
        resume_from=resume_from,
    )


def execute_stage2(
    config: PipelineConfig,
    plan: StagePlan,
    *,
    upstream_artifacts: Mapping[str, Path] | None = None,
) -> StageResult:
    return execute_stage(config, plan, upstream_artifacts=upstream_artifacts)


__all__ = ["execute_stage2", "plan_stage2"]
