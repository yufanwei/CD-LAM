"""Validated dependency boundary for external LAM and ACWM integrations."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, FrozenSet

from .config import StageName
from .training.common import StageContext, StageResult


class AdapterError(RuntimeError):
    """Raised when an external training adapter is absent or invalid."""


class StageAdapter(ABC):
    """Explicit interface implemented by external training integrations.

    CD-LAM intentionally does not guess an upstream LAM or ACWM call signature.
    Production integrations validate their dependencies and execute behind this
    boundary.
    """

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return a stable identity recorded in results and checkpoints."""

    @property
    @abstractmethod
    def supported_stages(self) -> FrozenSet[StageName]:
        """Return the stages this adapter explicitly supports."""

    @abstractmethod
    def validate(self, context: StageContext) -> None:
        """Fail before mutation if dependencies or contracts are invalid."""

    @abstractmethod
    def run(self, context: StageContext) -> StageResult:
        """Execute one validated stage and return auditable metadata."""


def load_stage_adapter(specification: str, stage: StageName) -> StageAdapter:
    """Load module:attribute and validate the resulting StageAdapter."""

    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise AdapterError(
            "adapter specification must use 'python.module:factory_or_instance'"
        )
    try:
        module = importlib.import_module(module_name)
        target: Any = getattr(module, attribute_name)
        adapter = (
            target()
            if callable(target) and not isinstance(target, StageAdapter)
            else target
        )
    except Exception as exc:
        raise AdapterError(f"failed to load adapter {specification!r}: {exc}") from exc
    if not isinstance(adapter, StageAdapter):
        raise AdapterError(
            f"adapter {specification!r} did not produce a StageAdapter instance"
        )
    if not adapter.identity.strip():
        raise AdapterError("adapter identity must be non-empty")
    if stage not in adapter.supported_stages:
        raise AdapterError(
            f"adapter {adapter.identity!r} does not support stage {stage.value!r}"
        )
    return adapter


__all__ = ["AdapterError", "StageAdapter", "load_stage_adapter"]
