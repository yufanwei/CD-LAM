"""Shared contracts and checkpoint metadata for stage runners."""

from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from ..config import PipelineConfig, StageName
from ..plans import StagePlan


class StageExecutionError(RuntimeError):
    """Raised when a validated stage cannot complete safely."""


@dataclass(frozen=True)
class StageContext:
    config: PipelineConfig
    plan: StagePlan
    upstream_artifacts: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    stage: StageName
    status: str
    config_digest: str
    seed: int
    start_step: int
    steps: int
    adapter_identity: str
    checkpoint: Path
    initial_loss: float
    final_loss: float
    best_loss: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    upstream_artifacts: Mapping[str, str] = field(default_factory=dict)
    upstream_hashes: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_identity": self.adapter_identity,
            "best_loss": self.best_loss,
            "checkpoint": str(self.checkpoint),
            "config_digest": self.config_digest,
            "final_loss": self.final_loss,
            "initial_loss": self.initial_loss,
            "metrics": dict(sorted(self.metrics.items())),
            "seed": self.seed,
            "stage": self.stage.value,
            "start_step": self.start_step,
            "status": self.status,
            "steps": self.steps,
            "upstream_artifacts": dict(sorted(self.upstream_artifacts.items())),
            "upstream_hashes": dict(sorted(self.upstream_hashes.items())),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def checkpoint_metadata(
    context: StageContext, *, completed_steps: int
) -> dict[str, Any]:
    """Return metadata required in every checkpoint produced by public runners."""

    return {
        "action_transform_id": context.plan.action_transform_id,
        "adapter_identity": context.plan.adapter_identity,
        "config_digest": context.plan.config_digest,
        "model_scale": context.plan.model_scale,
        "protocol_steps": context.plan.protocol_steps,
        "seed": context.plan.seed,
        "source_stride": context.plan.source_stride,
        "stage": context.plan.stage.value,
        "steps": completed_steps,
        "upstream_artifacts": {
            key: str(path) for key, path in sorted(context.upstream_artifacts.items())
        },
        "upstream_hashes": {
            key: file_sha256(path)
            for key, path in sorted(context.upstream_artifacts.items())
        },
    }


def file_sha256(path: Path) -> str:
    """Hash an upstream artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resume_checkpoint(
    context: StageContext,
) -> tuple[Optional[dict[str, Any]], int]:
    """Load and validate a synthetic checkpoint selected by a stage plan."""

    path = context.plan.resume_from
    if path is None:
        return None, 0
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise StageExecutionError(
            f"failed to load resume checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(blob, dict) or not isinstance(blob.get("metadata"), dict):
        raise StageExecutionError("resume checkpoint is missing metadata")
    metadata = blob["metadata"]
    expected = {
        "action_transform_id": context.plan.action_transform_id,
        "stage": context.plan.stage.value,
        "config_digest": context.plan.config_digest,
        "model_scale": context.plan.model_scale,
        "protocol_steps": context.plan.protocol_steps,
        "seed": context.plan.seed,
        "source_stride": context.plan.source_stride,
        "adapter_identity": context.plan.adapter_identity,
        "upstream_hashes": {
            key: file_sha256(path)
            for key, path in sorted(context.upstream_artifacts.items())
        },
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise StageExecutionError(
                f"resume checkpoint metadata mismatch for {key}: "
                f"expected {value!r}, got {metadata.get(key)!r}"
            )
    completed = metadata.get("steps")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise StageExecutionError(
            "resume checkpoint metadata.steps must be non-negative"
        )
    if completed >= context.plan.target_steps:
        raise StageExecutionError(
            f"resume checkpoint already has {completed} steps; "
            f"target is {context.plan.target_steps}"
        )
    return blob, completed


def assert_loss_trace(initial_loss: float, losses: list[float]) -> None:
    """Require finite optimization and a decreasing-or-bounded fixed-batch loss."""

    values = [initial_loss, *losses]
    if not losses or not all(math.isfinite(value) for value in values):
        raise StageExecutionError(f"non-finite synthetic loss trace: {values}")
    if min(losses) > initial_loss + 1e-6:
        raise StageExecutionError(
            f"synthetic optimizer never improved the fixed-batch loss: {values}"
        )
    bound = max(initial_loss * 1.25, initial_loss + 1e-4)
    if losses[-1] > bound:
        raise StageExecutionError(
            f"synthetic final loss exceeded stability bound {bound}: {values}"
        )


def make_result(
    context: StageContext,
    *,
    start_step: int,
    losses: list[float],
    initial_loss: float,
    metrics: Optional[Mapping[str, float]] = None,
) -> StageResult:
    assert_loss_trace(initial_loss, losses)
    return StageResult(
        stage=context.plan.stage,
        status="pass",
        config_digest=context.plan.config_digest,
        seed=context.plan.seed,
        start_step=start_step,
        steps=context.plan.target_steps,
        adapter_identity=context.plan.adapter_identity or "<missing>",
        checkpoint=context.plan.output_checkpoint,
        initial_loss=float(initial_loss),
        final_loss=float(losses[-1]),
        best_loss=float(min(losses)),
        metrics={} if metrics is None else dict(metrics),
        upstream_artifacts={
            key: str(path) for key, path in sorted(context.upstream_artifacts.items())
        },
        upstream_hashes={
            key: file_sha256(path)
            for key, path in sorted(context.upstream_artifacts.items())
        },
    )


def write_result_metadata(result: StageResult) -> Path:
    path = result.checkpoint.with_suffix(".result.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json() + "\n", encoding="utf-8")
    return path


def validate_result(result: StageResult, context: StageContext) -> None:
    if result.status != "pass":
        raise StageExecutionError(
            f"adapter returned non-pass status: {result.status!r}"
        )
    if result.stage != context.plan.stage:
        raise StageExecutionError("adapter returned a result for the wrong stage")
    if result.config_digest != context.plan.config_digest:
        raise StageExecutionError(
            "adapter returned a result for the wrong config digest"
        )
    if result.seed != context.plan.seed or result.steps != context.plan.target_steps:
        raise StageExecutionError("adapter returned inconsistent seed/step metadata")
    if result.adapter_identity != context.plan.adapter_identity:
        raise StageExecutionError("adapter identity changed between plan and result")
    if not all(
        math.isfinite(value)
        for value in (result.initial_loss, result.final_loss, result.best_loss)
    ):
        raise StageExecutionError("adapter returned non-finite loss metadata")
    if not result.checkpoint.is_file():
        raise StageExecutionError(
            f"adapter did not produce its declared checkpoint: {result.checkpoint}"
        )
    if context.plan.mode == "external":
        expected_paths = {
            key: str(path) for key, path in sorted(context.upstream_artifacts.items())
        }
        if dict(result.upstream_artifacts) != expected_paths:
            raise StageExecutionError(
                "external adapter returned incomplete or inconsistent upstream paths"
            )
        if set(result.upstream_hashes) != set(expected_paths):
            raise StageExecutionError(
                "external adapter must hash every configured upstream artifact"
            )
        invalid = {
            key: value
            for key, value in result.upstream_hashes.items()
            if not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        }
        if invalid:
            raise StageExecutionError(
                f"external adapter returned invalid upstream SHA-256 values: {invalid}"
            )


__all__ = [
    "StageContext",
    "StageExecutionError",
    "StageResult",
    "assert_loss_trace",
    "checkpoint_metadata",
    "file_sha256",
    "load_resume_checkpoint",
    "make_result",
    "validate_result",
    "write_result_metadata",
]
