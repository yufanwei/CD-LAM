from __future__ import annotations

import re
from pathlib import Path

import yaml

from cdlam_runtime.resolve_stage1 import resolve_config


ROOT = Path(__file__).resolve().parents[1]


def test_public_stage1_recipe_hides_compatibility_version_names(tmp_path: Path) -> None:
    template = ROOT / "configs" / "stage1_recipe.yaml"
    text = template.read_text(encoding="utf-8")
    assert re.search(r"\bv[0-9]", text, flags=re.IGNORECASE) is None

    resolved = resolve_config(
        template,
        tmp_path / "train.parquet",
        tmp_path / "resolved.yaml",
        eval_pair_index=tmp_path / "eval.parquet",
    )
    document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    loss = document["trainer"]["loss"]
    cadence = document["trainer"]["cadence"]
    assert "masked_reconstruction" not in loss
    assert "contrastive_extensions" not in loss
    assert loss["v5_partial_fullmix"] is True
    assert loss["v5_gamma_bg"] == 0.02
    assert loss["siglip_use_v42_graph"] is False
    assert "baseline_checkpoint" not in cadence
    assert cadence["f1_v3_baseline_ckpt"] == ""


def test_stage1_artifacts_use_semantic_trainer_names() -> None:
    trainer = (
        ROOT
        / "third_party"
        / "acwm_overlay"
        / "training_scope"
        / "LAM"
        / "V5"
        / "tools"
        / "train_lam_v5.py"
    ).read_text(encoding="utf-8")
    assert '"trainer": "cdlam_stage1"' in trainer
    assert '"cdlam_stage1_siglip"' in trainer
    assert '"cdlam_stage1_pairwise"' in trainer
    assert "v3_routeA_v0" not in trainer
    assert "v4_routeA_v4_1" not in trainer
