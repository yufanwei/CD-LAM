"""Atomically bind a newly trained bridge to the validated action contract."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from cdlam_runtime.action_contract import (
    ACTION_DIM,
    ACTION_TOKENS,
    BLOCK_SIZE,
    BRIDGE_REPRESENTATION,
    LOADER_REPRESENTATION,
    SOURCE_STRIDE,
    ActionContractError,
    load_stage3_action_contract,
    sha256_file,
)


def _read_contract(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionContractError(f"invalid action contract {path}: {exc}") from exc
    if not isinstance(document, dict) or not document.get("contract_id"):
        raise ActionContractError("action contract must contain a contract_id")
    return document


def bind_bridge_contract(
    checkpoint: str | Path,
    contract_path: str | Path,
    external_root: str | Path,
    stage1_checkpoint: str | Path,
    *,
    lineage: str,
) -> dict[str, Any]:
    """Write contract metadata only after a same-directory candidate validates."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    contract_path = Path(contract_path).expanduser().resolve()
    external_root = Path(external_root).expanduser().resolve()
    stage1_checkpoint = Path(stage1_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ActionContractError(f"bridge checkpoint is missing: {checkpoint}")
    if not stage1_checkpoint.is_file():
        raise ActionContractError(f"Stage-1 checkpoint is missing: {stage1_checkpoint}")

    contract = _read_contract(contract_path)
    stats_path = external_root / str(contract.get("stats_relative_path", ""))
    modality_path = external_root / str(contract.get("modality_relative_path", ""))
    stats_digest = sha256_file(stats_path)
    modality_digest = sha256_file(modality_path)
    if stats_digest != contract.get("stats_sha256"):
        raise ActionContractError("AgiBot statistics do not match the action contract")
    if modality_digest != contract.get("modality_sha256"):
        raise ActionContractError(
            "AgiBot modality metadata do not match the action contract"
        )

    try:
        bridge = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        bridge = torch.load(checkpoint, map_location="cpu")
    if not isinstance(bridge, dict):
        raise ActionContractError("bridge checkpoint must contain a dictionary")
    metadata = {
        "contract_id": contract["contract_id"],
        "action_dim": ACTION_DIM,
        "action_tokens": ACTION_TOKENS,
        "block_size": BLOCK_SIZE,
        "source_stride": SOURCE_STRIDE,
        "loader_representation": LOADER_REPRESENTATION,
        "representation": BRIDGE_REPRESENTATION,
        "scale_family": lineage,
        "stats_sha256": stats_digest,
        "modality_sha256": modality_digest,
        "stage1_checkpoint_sha256": sha256_file(stage1_checkpoint),
    }
    existing = bridge.get("cdlam_action_contract")
    if existing is not None and existing != metadata:
        raise ActionContractError(
            "bridge already contains different action-contract metadata"
        )
    bridge["cdlam_action_contract"] = metadata

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{checkpoint.name}.",
        suffix=".tmp",
        dir=checkpoint.parent,
        delete=False,
    ) as handle:
        candidate = Path(handle.name)
    try:
        torch.save(bridge, candidate)
        load_stage3_action_contract(
            contract_path,
            external_root,
            candidate,
            lineage,
        )
        os.replace(candidate, checkpoint)
    finally:
        candidate.unlink(missing_ok=True)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--lineage", default="100h")
    args = parser.parse_args()
    metadata = bind_bridge_contract(
        args.checkpoint,
        args.contract,
        args.external_root,
        args.stage1_checkpoint,
        lineage=args.lineage,
    )
    print(json.dumps({"status": "ok", **metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
