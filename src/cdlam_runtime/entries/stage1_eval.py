"""Stage-1 evaluation adapter for embedded shard bytes and portable paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ["CDLAM_ACWM_ROOT"]).resolve()
PYTHON_ROOT = Path(os.environ.get("CDLAM_PYTHON_ROOT", str(REPO.parent))).resolve()
for path in (
    PYTHON_ROOT,
    REPO,
):
    sys.path.insert(0, str(path))

from Scale.common.shard_io import decode_rows_from_shard  # noqa: E402
from cdlam_integration.lam import eval as _evalmod  # noqa: E402
from cdlam_integration.lam import eval_protocol as _evalfull  # noqa: E402
from cdlam_integration.lam import model_loader as _lambuilder  # noqa: E402

LAM_BASE = Path(os.environ["LAM_400K_LOCAL"]).resolve()


def _build_lam(name: str = "LAM", device: str = "cuda"):
    """Load the configured unmodified base LAM."""

    _lambuilder._patch_sdpa()
    from external.lam.model import LAM as model_class
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
