from __future__ import annotations

import sys
import types
from dataclasses import replace
from typing import FrozenSet

import pytest

from cd_lam.adapters import AdapterError, StageAdapter, load_stage_adapter
from cd_lam.config import PipelineConfig, StageName
from cd_lam.plans import build_stage_plan
from cd_lam.training.common import (
    StageContext,
    StageExecutionError,
    StageResult,
    validate_result,
)


class _DummyAdapter(StageAdapter):
    @property
    def identity(self) -> str:
        return "tests.dummy_adapter"

    @property
    def supported_stages(self) -> FrozenSet[StageName]:
        return frozenset({StageName.STAGE2})

    def validate(self, context: StageContext) -> None:
        del context

    def run(self, context: StageContext) -> StageResult:
        raise AssertionError("loader test does not execute the adapter")


def test_dynamic_adapter_loader_validates_identity_and_stage(monkeypatch) -> None:
    module = types.ModuleType("cd_lam_test_adapter")
    module.factory = _DummyAdapter
    monkeypatch.setitem(sys.modules, module.__name__, module)

    adapter = load_stage_adapter("cd_lam_test_adapter:factory", StageName.STAGE2)
    assert adapter.identity == "tests.dummy_adapter"
    with pytest.raises(AdapterError, match="does not support"):
        load_stage_adapter("cd_lam_test_adapter:factory", StageName.STAGE3)


def test_adapter_loader_fails_closed_on_arbitrary_object(monkeypatch) -> None:
    module = types.ModuleType("cd_lam_bad_adapter")
    module.value = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(AdapterError, match="StageAdapter"):
        load_stage_adapter("cd_lam_bad_adapter:value", StageName.STAGE1)


def test_external_result_requires_parent_artifact_hashes(tmp_path) -> None:
    config = PipelineConfig.synthetic(tmp_path / "outputs")
    plan = build_stage_plan(config, StageName.STAGE1, synthetic=True, target_steps=1)
    plan = replace(plan, mode="external", adapter_identity="tests.external")
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    context = StageContext(config, plan, {"lam_init": parent})
    result = StageResult(
        stage=StageName.STAGE1,
        status="pass",
        config_digest=config.digest,
        seed=plan.seed,
        start_step=0,
        steps=1,
        adapter_identity="tests.external",
        checkpoint=checkpoint,
        initial_loss=1.0,
        final_loss=0.5,
        best_loss=0.5,
        upstream_artifacts={"lam_init": str(parent)},
        upstream_hashes={},
    )
    with pytest.raises(StageExecutionError, match="hash every"):
        validate_result(result, context)
