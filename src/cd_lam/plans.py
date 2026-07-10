"""Deterministic dry-run plans for CD-LAM training stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import BaseStageConfig, PipelineConfig, StageName


class PlanError(ValueError):
    """Raised when command-line planning arguments are invalid."""


@dataclass(frozen=True)
class AssetRequirement:
    name: str
    path: Optional[Path]
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "name": self.name,
            "path": None if self.path is None else str(self.path),
        }


@dataclass(frozen=True)
class StagePlan:
    stage: StageName
    mode: str
    config_name: str
    config_digest: str
    model_scale: str
    protocol_steps: int
    target_steps: int
    batch_size: int
    learning_rate: float
    seed: int
    device: str
    adapter_specification: Optional[str]
    adapter_identity: Optional[str]
    required_assets: tuple[AssetRequirement, ...]
    output_checkpoint: Path
    resume_from: Optional[Path]
    working_directory: Path
    action_transform_id: Optional[str]
    source_stride: Optional[int]
    blockers: tuple[str, ...]
    actions: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible plan."""

        return {
            "actions": list(self.actions),
            "action_transform_id": self.action_transform_id,
            "adapter_identity": self.adapter_identity,
            "adapter_specification": self.adapter_specification,
            "batch_size": self.batch_size,
            "blockers": list(self.blockers),
            "config_digest": self.config_digest,
            "config_name": self.config_name,
            "device": self.device,
            "learning_rate": self.learning_rate,
            "mode": self.mode,
            "model_scale": self.model_scale,
            "output_checkpoint": str(self.output_checkpoint),
            "protocol_steps": self.protocol_steps,
            "ready": self.ready,
            "required_assets": [item.to_dict() for item in self.required_assets],
            "resume_from": None if self.resume_from is None else str(self.resume_from),
            "seed": self.seed,
            "source_stride": self.source_stride,
            "stage": self.stage.value,
            "target_steps": self.target_steps,
            "working_directory": str(self.working_directory),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


_OUTPUT_NAMES = {
    StageName.STAGE1: ("stage1", "lam_debiased.pt"),
    StageName.STAGE2: ("stage2", "acwm_debiased.pt"),
    StageName.BRIDGE: ("bridge", "action_to_latent.pt"),
    StageName.STAGE3: ("stage3", "acwm_robot_action.pt"),
}


def _asset_requirements(config: PipelineConfig, stage: StageName) -> tuple[AssetRequirement, ...]:
    paths = config.paths
    selected: tuple[tuple[str, Optional[Path]], ...] = {
        StageName.STAGE1: (
            ("unlabeled_manifest", paths.unlabeled_manifest),
            ("lam_init", paths.lam_init),
        ),
        StageName.STAGE2: (
            ("unlabeled_manifest", paths.unlabeled_manifest),
            ("base_acwm", paths.base_acwm),
            ("stage1_lam", paths.stage1_lam),
        ),
        StageName.BRIDGE: (
            ("robot_action_manifest", paths.robot_action_manifest),
            ("stage1_lam", paths.stage1_lam),
        ),
        StageName.STAGE3: (
            ("robot_action_manifest", paths.robot_action_manifest),
            ("stage2_acwm", paths.stage2_acwm),
            ("bridge_bundle", paths.bridge_bundle),
        ),
    }[stage]
    return tuple(
        AssetRequirement(name=name, path=path, exists=bool(path and path.exists()))
        for name, path in selected
    )


def _checkpoint_blockers(stage: BaseStageConfig) -> list[str]:
    blockers: list[str] = []
    observed = stage.observed_checkpoint_steps
    if (
        observed is not None
        and observed != stage.protocol_steps
        and not stage.allow_partial_checkpoint
    ):
        blockers.append(
            "observed checkpoint steps "
            f"({observed}) do not match declared protocol steps ({stage.protocol_steps}); "
            "set allow_partial_checkpoint=true only for an intentional resume"
        )
    return blockers


def _transform_blockers(config: PipelineConfig) -> list[str]:
    bridge = config.bridge
    stage3 = config.stage3
    blockers: list[str] = []
    if not bridge.action_transform_id:
        blockers.append("bridge_training.action_transform_id is required in real mode")
    if not stage3.action_transform_id:
        blockers.append("stage3.action_transform_id is required in real mode")
    if bridge.source_stride is None or bridge.source_stride < 1:
        blockers.append("bridge_training.source_stride must be a positive integer")
    if stage3.source_stride is None or stage3.source_stride < 1:
        blockers.append("stage3.source_stride must be a positive integer")
    if (
        bridge.action_transform_id
        and stage3.action_transform_id
        and bridge.action_transform_id != stage3.action_transform_id
    ):
        blockers.append(
            "bridge/Stage3 action_transform_id mismatch: "
            f"{bridge.action_transform_id!r} != {stage3.action_transform_id!r}"
        )
    if (
        bridge.source_stride is not None
        and stage3.source_stride is not None
        and bridge.source_stride != stage3.source_stride
    ):
        blockers.append(
            "bridge/Stage3 source_stride mismatch: "
            f"{bridge.source_stride} != {stage3.source_stride}"
        )
    return blockers


def build_stage_plan(
    config: PipelineConfig,
    stage: StageName,
    *,
    synthetic: bool = False,
    target_steps: Optional[int] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    adapter: Optional[str] = None,
    resume_from: Optional[Path] = None,
) -> StagePlan:
    """Build a fail-closed execution plan without mutating the filesystem."""

    stage_config = config.stage(stage)
    steps = stage_config.protocol_steps if target_steps is None else target_steps
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise PlanError("target_steps must be a positive integer")
    selected_seed = config.runtime.seed if seed is None else seed
    if isinstance(selected_seed, bool) or not isinstance(selected_seed, int) or selected_seed < 0:
        raise PlanError("seed must be a non-negative integer")
    selected_device = device or ("cpu" if synthetic else config.runtime.device)
    if synthetic and selected_device != "cpu":
        raise PlanError("synthetic training is CPU-only; use --device cpu")

    mode = "synthetic" if synthetic else "external"
    adapter_specification = None if synthetic else (adapter or stage_config.adapter)
    adapter_identity = "cd_lam.synthetic_cpu" if synthetic else None
    required_assets = () if synthetic else _asset_requirements(config, stage)
    blockers = _checkpoint_blockers(stage_config)
    selected_resume = (resume_from or stage_config.resume_from)
    if selected_resume is not None:
        selected_resume = selected_resume.expanduser().resolve()
        if not selected_resume.is_file():
            blockers.append(f"resume checkpoint does not exist: {selected_resume}")

    if not synthetic:
        if not adapter_specification:
            adapter_key = (
                "adapters.bridge or adapters.bridge_train"
                if stage == StageName.BRIDGE
                else f"adapters.{stage.value}"
            )
            blockers.append(
                f"{stage.value} has no external adapter; configure {adapter_key} "
                "or pass --adapter"
            )
        for requirement in required_assets:
            if requirement.path is None:
                blockers.append(f"required asset is not configured: {requirement.name}")
            elif not requirement.exists:
                blockers.append(
                    f"required asset does not exist: {requirement.name}={requirement.path}"
                )
        if stage in {StageName.BRIDGE, StageName.STAGE3}:
            blockers.extend(_transform_blockers(config))
        if stage == StageName.STAGE3:
            if config.stage3.working_directory is None:
                blockers.append("stage3.working_directory is required for an external adapter")
            elif not config.stage3.working_directory.is_dir():
                blockers.append(
                    "stage3.working_directory is not a directory: "
                    f"{config.stage3.working_directory}"
                )

    directory, filename = _OUTPUT_NAMES[stage]
    output_checkpoint = (config.paths.output_root / directory / filename).resolve()
    working_directory = (
        config.stage3.working_directory
        if stage == StageName.STAGE3 and config.stage3.working_directory is not None
        else config.paths.project_root
    )
    action_transform_id = None
    source_stride = None
    if stage == StageName.BRIDGE:
        action_transform_id = config.bridge.action_transform_id
        source_stride = config.bridge.source_stride
    elif stage == StageName.STAGE3:
        action_transform_id = config.stage3.action_transform_id
        source_stride = config.stage3.source_stride

    actions = (
        f"load typed config {config.name} ({config.digest[:12]})",
        f"initialize {adapter_identity or adapter_specification or '<missing adapter>'} on {selected_device}",
        f"run {steps} total optimizer step(s)",
        f"write checkpoint {output_checkpoint}",
        f"write result metadata {output_checkpoint.with_suffix('.result.json')}",
    )
    return StagePlan(
        stage=stage,
        mode=mode,
        config_name=config.name,
        config_digest=config.digest,
        model_scale=config.model_scale,
        protocol_steps=stage_config.protocol_steps,
        target_steps=steps,
        batch_size=stage_config.batch_size,
        learning_rate=stage_config.learning_rate,
        seed=selected_seed,
        device=selected_device,
        adapter_specification=adapter_specification,
        adapter_identity=adapter_identity,
        required_assets=required_assets,
        output_checkpoint=output_checkpoint,
        resume_from=selected_resume,
        working_directory=working_directory.resolve(),
        action_transform_id=action_transform_id,
        source_stride=source_stride,
        blockers=tuple(dict.fromkeys(blockers)),
        actions=actions,
    )


__all__ = [
    "AssetRequirement",
    "PlanError",
    "StagePlan",
    "build_stage_plan",
]
