"""Model-agnostic bridge and no-bridge conditioning example."""

from __future__ import annotations

import torch

from cd_lam import (
    ACTION_DIM,
    LATENT_DIM,
    ActionToLatentBridge,
    build_bridge_mlp,
    prepare_latent_condition,
)


def main() -> None:
    # No-bridge route: use an already encoded 32D latent action.
    encoded_latent = torch.zeros(2, 4, LATENT_DIM)
    latent_condition = prepare_latent_condition(latent=encoded_latent)

    # Bridge route: a real release loads this complete bundle from a trusted
    # checkpoint.  Identity statistics make the example self-contained.
    synthetic_bundle = {
        "g_state": build_bridge_mlp().state_dict(),
        "action_mean": torch.zeros(ACTION_DIM),
        "action_std": torch.ones(ACTION_DIM),
        "zm": torch.zeros(LATENT_DIM),
        "zsd": torch.ones(LATENT_DIM),
        "latent_dim": LATENT_DIM,
    }
    bridge = ActionToLatentBridge(synthetic_bundle)
    robot_action = torch.zeros(2, 4, ACTION_DIM)
    bridged_condition = prepare_latent_condition(
        robot_action=robot_action,
        bridge=bridge,
    )

    print("no bridge:", tuple(latent_condition.shape), latent_condition.dtype)
    print("with bridge:", tuple(bridged_condition.shape), bridged_condition.dtype)


if __name__ == "__main__":
    main()
