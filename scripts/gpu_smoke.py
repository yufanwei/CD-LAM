#!/usr/bin/env python3
"""Run a small real CUDA optimizer step for the unified CD-LAM environment."""

from __future__ import annotations

import argparse
import json
import math

import torch
from torch import nn


EXPECTED_TORCH = "2.7.0+cu128"
EXPECTED_CUDA = "12.8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu < 0:
        raise SystemExit("--gpu must be non-negative")
    if torch.__version__ != EXPECTED_TORCH:
        raise SystemExit(f"expected torch {EXPECTED_TORCH}, found {torch.__version__}")
    if torch.version.cuda != EXPECTED_CUDA:
        raise SystemExit(
            f"expected CUDA wheel {EXPECTED_CUDA}, found {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")
    if args.gpu >= torch.cuda.device_count():
        raise SystemExit(
            f"GPU index {args.gpu} is outside {torch.cuda.device_count()} visible device(s)"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda", args.gpu)
    properties = torch.cuda.get_device_properties(device)
    if properties.major not in {8, 9}:
        raise SystemExit(
            "CD-LAM supports Ampere or Hopper GPUs; found compute capability "
            f"{properties.major}.{properties.minor}"
        )
    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(16, 32, device=device)
    targets = torch.randn(16, 32, device=device)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    if not bool(torch.isfinite(loss)):
        raise SystemExit("CUDA optimizer smoke produced a non-finite loss")
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(value).all()) for value in gradients
    ):
        raise SystemExit(
            "CUDA optimizer smoke produced missing or non-finite gradients"
        )
    optimizer.step()
    torch.cuda.synchronize(device)

    changed = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters(), strict=True)
    )
    if not changed or not math.isfinite(float(loss.detach())):
        raise SystemExit("CUDA optimizer smoke did not update model parameters")

    print(
        json.dumps(
            {
                "cuda": torch.version.cuda,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "device": args.gpu,
                "device_name": properties.name,
                "loss": float(loss.detach()),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                "status": "pass",
                "torch": torch.__version__,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
