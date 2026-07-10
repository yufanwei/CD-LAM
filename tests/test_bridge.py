from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from cd_lam.bridge import (
    ACTION_DIM,
    LATENT_DIM,
    ActionToLatentBridge,
    BridgeCheckpointError,
    build_bridge_mlp,
    prepare_latent_condition,
    validate_bridge_checkpoint,
)


def synthetic_checkpoint() -> dict:
    torch.manual_seed(7)
    model = build_bridge_mlp()
    return {
        "g_state": OrderedDict((key, value.detach().clone()) for key, value in model.state_dict().items()),
        "action_mean": torch.linspace(-1.0, 1.0, ACTION_DIM),
        "action_std": torch.linspace(0.5, 1.5, ACTION_DIM),
        "zm": torch.linspace(-0.2, 0.2, LATENT_DIM),
        "zsd": torch.linspace(0.8, 1.2, LATENT_DIM),
        "latent_dim": LATENT_DIM,
        "note": "synthetic public-contract test",
    }


def test_bridge_synthetic_checkpoint_file_roundtrip(tmp_path) -> None:
    checkpoint = synthetic_checkpoint()
    path = tmp_path / "bridge.pt"
    torch.save(checkpoint, path)

    bridge = ActionToLatentBridge.from_checkpoint(path)
    action = torch.linspace(-2.0, 2.0, 2 * 3 * ACTION_DIM).reshape(2, 3, ACTION_DIM)
    expected_normalized = (action - checkpoint["action_mean"]) / checkpoint["action_std"]
    expected = build_bridge_mlp()
    expected.load_state_dict(checkpoint["g_state"], strict=True)
    expected_latent = expected(expected_normalized) * checkpoint["zsd"] + checkpoint["zm"]

    actual = bridge(action)
    assert actual.shape == (2, 3, LATENT_DIM)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected_latent)
    assert not any(parameter.requires_grad for parameter in bridge.parameters())

    restored = ActionToLatentBridge(bridge.to_checkpoint())
    torch.testing.assert_close(restored(action), actual)
    assert restored.metadata["note"] == checkpoint["note"]


def test_bridge_accepts_one_action_vector() -> None:
    bridge = ActionToLatentBridge(synthetic_checkpoint())
    assert bridge(torch.zeros(ACTION_DIM)).shape == (LATENT_DIM,)


@pytest.mark.parametrize("missing", ["g_state", "action_mean", "action_std", "zm", "zsd", "latent_dim"])
def test_bridge_validator_requires_full_contract(missing: str) -> None:
    checkpoint = synthetic_checkpoint()
    del checkpoint[missing]
    with pytest.raises(BridgeCheckpointError, match=missing):
        validate_bridge_checkpoint(checkpoint)


def test_bridge_validator_rejects_wrong_architecture_and_scale() -> None:
    checkpoint = synthetic_checkpoint()
    checkpoint["g_state"]["0.weight"] = torch.zeros(128, ACTION_DIM)
    with pytest.raises(BridgeCheckpointError, match="0.weight.*shape"):
        validate_bridge_checkpoint(checkpoint)

    checkpoint = synthetic_checkpoint()
    checkpoint["action_std"][3] = 0
    with pytest.raises(BridgeCheckpointError, match="action_std.*strictly positive"):
        validate_bridge_checkpoint(checkpoint)


def test_bridge_reports_bad_action_shape_and_values() -> None:
    bridge = ActionToLatentBridge(synthetic_checkpoint())
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 22\)"):
        bridge(torch.zeros(2, 21))
    bad = torch.zeros(1, ACTION_DIM)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        bridge(bad)


def test_prepare_latent_condition_supports_both_exclusive_routes() -> None:
    direct = torch.randn(2, 4, LATENT_DIM, dtype=torch.float64)
    prepared_direct = prepare_latent_condition(latent=direct)
    assert prepared_direct.shape == direct.shape
    assert prepared_direct.dtype == torch.float32
    assert prepared_direct.device == direct.device
    torch.testing.assert_close(prepared_direct, direct.float())

    bridge = ActionToLatentBridge(synthetic_checkpoint())
    action = torch.randn(2, 4, ACTION_DIM)
    prepared_action = prepare_latent_condition(robot_action=action, bridge=bridge)
    torch.testing.assert_close(prepared_action, bridge(action))
    assert prepared_action.shape == (2, 4, LATENT_DIM)
    assert prepared_action.dtype == torch.float32


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "exactly one"),
        ({"latent": torch.zeros(LATENT_DIM), "robot_action": torch.zeros(ACTION_DIM)}, "exactly one"),
        ({"robot_action": torch.zeros(ACTION_DIM)}, "provided together"),
        ({"bridge": object()}, "provided together"),
        (
            {"robot_action": torch.zeros(ACTION_DIM), "bridge": object()},
            "ActionToLatentBridge",
        ),
        ({"latent": torch.zeros(31)}, r"shape \(\.\.\., 32\)"),
    ],
)
def test_prepare_latent_condition_rejects_ambiguous_or_invalid_routes(kwargs, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        prepare_latent_condition(**kwargs)


def test_prepare_latent_condition_rejects_nonfinite_latent() -> None:
    latent = torch.zeros(LATENT_DIM)
    latent[0] = torch.inf
    with pytest.raises(ValueError, match="NaN or infinity"):
        prepare_latent_condition(latent=latent)
