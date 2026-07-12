"""Stage-1 evaluation adapter for embedded shard bytes and portable paths."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO = Path(os.environ["CDLAM_ACWM_ROOT"]).resolve()
PYTHON_ROOT = Path(os.environ.get("CDLAM_PYTHON_ROOT", str(REPO.parent))).resolve()
for path in (
    PYTHON_ROOT,
    REPO,
    REPO / "training_scope" / "LAM",
    REPO / "finetune_4-30" / "scripts",
):
    sys.path.insert(0, str(path))

for name in ("V2", "V3", "V4", "V5", "V6", "V6_1", "V7"):
    sys.modules.setdefault(f"LAM_{name}", importlib.import_module(name))
    sys.modules.setdefault(
        f"LAM_{name}.tools", importlib.import_module(f"{name}.tools")
    )

decode_rows_from_shard = importlib.import_module(
    "Scale.common.shard_io"
).decode_rows_from_shard
_lambuilder = importlib.import_module("LAM_V2.tools.train_lam_action_readout")
_evalmod = importlib.import_module("LAM_V2.tools.eval_lam_v2_g1")
_evalfull = importlib.import_module("LAM_V2.tools.eval_lam_v2_full")

LAM_BASE = Path(os.environ["LAM_400K_LOCAL"]).resolve()


def _build_lam(name: str = "LAM", device: str = "cuda"):
    """Load the configured unmodified base LAM."""

    _lambuilder._patch_sdpa()
    model_class = importlib.import_module("external.lam.model").LAM
    model = model_class(
        image_channels=3,
        lam_model_dim=1024,
        lam_latent_dim=32,
        lam_patch_size=16,
        lam_enc_blocks=24,
        lam_dec_blocks=24,
        lam_num_heads=16,
        ckpt_path=str(LAM_BASE),
    )
    return model.to(device).eval()


def _decode_pairs(rows, target_hw=(240, 320), workers=16):
    return decode_rows_from_shard(rows, target_hw, workers)


def main() -> int:
    for module in (_evalmod, _evalfull, _lambuilder):
        if hasattr(module, "build_lam"):
            module.build_lam = _build_lam
        if hasattr(module, "decode_pairs_parallel"):
            module.decode_pairs_parallel = _decode_pairs
    print(
        f"[Stage-1 eval] base={LAM_BASE}; frame_source=embedded_shard_bytes",
        flush=True,
    )
    return int(_evalmod.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
