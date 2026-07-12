#!/usr/bin/env python3
"""Contract-safe adapter for the external CD-LAM Stage-3 trainer.

This entry validates the legacy bridge and AgiBot metadata on CPU, converts
the loader's normalized block-anchor deltas before every bridge call, and
runs the external trainer from its configured repository root. The external
research tree remains unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import torch

from cdlam_runtime.action_contract import (
    ActionContractError,
    Stage3ActionContract,
    load_stage3_action_contract,
)


def _argument_value(
    argv: Sequence[str],
    flag: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == flag:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ActionContractError(f"{flag} requires a value")
            values.append(argv[index + 1])
        elif token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
    if len(values) > 1:
        raise ActionContractError(f"{flag} was provided more than once")
    if values:
        return values[0]
    if required:
        raise ActionContractError(f"Stage 3 requires an explicit {flag}")
    return default


def _external_root() -> Path:
    value = os.environ.get("CDLAM_ACWM_ROOT")
    if not value:
        raise ActionContractError(
            "CDLAM_ACWM_ROOT must identify the external repository"
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ActionContractError(f"external ACWM repository is missing: {root}")
    return root


def _load_external_stage3(external_root: Path) -> ModuleType:
    entry = external_root / "New LAM" / "Post Train" / "train_gbridge_z_posttrain.py"
    if not entry.is_file():
        raise ActionContractError(f"external Stage-3 entry is missing: {entry}")

    for path in (external_root, external_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    spec = importlib.util.spec_from_file_location("cdlam_external_stage3", entry)
    if spec is None or spec.loader is None:
        raise ActionContractError(f"unable to create an import spec for {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_contract_bridge(
    external: ModuleType,
    contract: Stage3ActionContract,
) -> None:
    original_bridge = external.FrozenBridge

    class ContractBridge(original_bridge):
        """External frozen bridge with an enforced loader-to-bridge conversion."""

        def __init__(self, checkpoint_path, device):
            observed = Path(checkpoint_path).expanduser().resolve()
            if observed != contract.bridge_path:
                raise ActionContractError(
                    f"runtime bridge changed after validation: {observed} != {contract.bridge_path}"
                )
            super().__init__(checkpoint_path, device)
            self.meta["cdlam_action_contract"] = contract.summary()

        def z_from_action(self, raw_adjacent):
            contract.validate_bridge_input(raw_adjacent)
            return super().z_from_action(raw_adjacent)

    ContractBridge.__name__ = "ContractBridge"
    ContractBridge.__qualname__ = "ContractBridge"
    external.FrozenBridge = ContractBridge


def _convert_dataset_item(
    item: dict,
    contract: Stage3ActionContract,
    source_slice: tuple[int, int] = (147, 169),
) -> dict:
    if not isinstance(item, dict):
        raise ActionContractError(
            f"dataset item must be a dictionary, got {type(item)!r}"
        )
    action = item.get("action")
    if not isinstance(action, torch.Tensor):
        raise ActionContractError("dataset item has no action tensor")
    if tuple(action.shape) != (12, 384):
        raise ActionContractError(
            f"dataset action shape is {tuple(action.shape)}, expected (12, 384)"
        )
    if action.device.type != "cpu" or action.dtype != torch.float32:
        raise ActionContractError(
            "dataset action must be converted on CPU in float32 before the trainer casts it"
        )
    start, end = source_slice
    if end - start != 22:
        raise ActionContractError(f"AgiBot source slice {source_slice} is not 22D")

    converted = dict(item)
    converted_action = action.clone()
    converted_action[..., start:end] = contract.loader_to_bridge(action[..., start:end])
    converted["action"] = converted_action
    return converted


def _install_contract_dataset(
    external: ModuleType,
    contract: Stage3ActionContract,
) -> None:
    source_slice = tuple(external.SLICE_BY_EMBODIMENT.get("agibot", ()))
    if source_slice != (147, 169):
        raise ActionContractError(
            f"external AgiBot action slice is {source_slice}, expected (147, 169)"
        )

    import groot_dreams.dataloader as dataloader

    original_dataset = dataloader.MultiVideoActionDataset

    class ContractDataset(original_dataset):
        """Dataset adapter that converts actions before the bfloat16 device cast."""

        def __getitem__(self, index):
            return _convert_dataset_item(
                super().__getitem__(index),
                contract,
                source_slice,
            )

    name = "CDLAMContractMultiVideoActionDataset"
    ContractDataset.__name__ = name
    ContractDataset.__qualname__ = name
    ContractDataset.__module__ = dataloader.__name__
    setattr(dataloader, name, ContractDataset)
    dataloader.MultiVideoActionDataset = ContractDataset


def main() -> int:
    external_root = _external_root()
    embodiment = _argument_value(sys.argv[1:], "--embodiment", default="agibot")
    if embodiment != "agibot":
        raise ActionContractError(
            f"this Stage-3 action contract supports only agibot, got {embodiment!r}"
        )
    bridge_value = _argument_value(
        sys.argv[1:],
        "--gr-bridge-ckpt",
        required=True,
    )
    assert bridge_value is not None
    scale_family = os.environ.get("CDLAM_SCALE_FAMILY")
    if not scale_family:
        raise ActionContractError("CDLAM_SCALE_FAMILY is required for bridge binding")
    contract_value = os.environ.get("CDLAM_STAGE3_ACTION_CONTRACT")
    if not contract_value:
        raise ActionContractError("CDLAM_STAGE3_ACTION_CONTRACT is required")
    contract_path = Path(contract_value)
    contract = load_stage3_action_contract(
        contract_path,
        external_root,
        bridge_value,
        scale_family,
    )
    print(
        "[CD-LAM Stage 3] validated action contract "
        + json.dumps(contract.summary(), sort_keys=True),
        flush=True,
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(external_root)
        external = _load_external_stage3(external_root)
        _install_contract_dataset(external, contract)
        _install_contract_bridge(external, contract)
        external.main()
    finally:
        os.chdir(previous_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
