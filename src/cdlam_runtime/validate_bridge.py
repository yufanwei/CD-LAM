#!/usr/bin/env python3
"""Strict CPU validation for the CD-LAM 22D-to-32D bridge bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REQUIRED = ("g_state", "action_mean", "action_std", "zm", "zsd", "latent_dim")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    path = Path(args.checkpoint)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in REQUIRED if key not in blob]
    if missing:
        raise SystemExit(f"bridge missing fields: {missing}")
    action_mean = np.asarray(blob["action_mean"], dtype=np.float32)
    action_std = np.asarray(blob["action_std"], dtype=np.float32)
    zm = np.asarray(blob["zm"], dtype=np.float32)
    zsd = np.asarray(blob["zsd"], dtype=np.float32)
    latent_dim = int(blob["latent_dim"])
    shapes = {
        "action_mean": list(action_mean.shape),
        "action_std": list(action_std.shape),
        "zm": list(zm.shape),
        "zsd": list(zsd.shape),
    }
    if shapes != {"action_mean": [22], "action_std": [22], "zm": [32], "zsd": [32]}:
        raise SystemExit(f"bridge shape mismatch: {shapes}")
    if latent_dim != 32:
        raise SystemExit(f"bridge latent_dim={latent_dim}, expected 32")
    if not all(
        np.isfinite(value).all() for value in (action_mean, action_std, zm, zsd)
    ):
        raise SystemExit("bridge statistics contain non-finite values")
    state = blob["g_state"]
    if tuple(state["0.weight"].shape) != (256, 22) or tuple(
        state["4.weight"].shape
    ) != (32, 256):
        raise SystemExit("bridge MLP dimensions are not 22D-to-32D")
    result = {
        "status": "ok",
        "checkpoint": str(path),
        "bytes": path.stat().st_size,
        "action_dim": 22,
        "latent_dim": latent_dim,
        "fields": list(REQUIRED),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
