"""Stage-1 masked LAM training entry point, executed once per torchrun rank.

The entry point fine-tunes the configured latent-action model with the
reference crop/scale path and mask-supervised reconstruction. It performs three
integration tasks before entering the external training loop:

1. Decode frame pairs from bytes embedded in an 18-column shard through
   ``decode_rows_from_shard`` instead of opening standalone MP4 files.
2. Install the ``frame_to_mask_idx``-aware mask loader. Rows without populated
   mask-cache entries fall back to full-frame reconstruction.
3. Start the external LAM training loop.

Input records come from ``lam_pair_index.parquet``, materialized from the data
recipe by ``Scale/common/build_scale_manifests.py``.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

try:
    REPO = Path(os.environ["CDLAM_ACWM_ROOT"]).expanduser().resolve()
except KeyError as exc:
    raise RuntimeError("CDLAM_ACWM_ROOT is required") from exc
PYTHON_ROOT = Path(os.environ.get("CDLAM_PYTHON_ROOT", str(REPO.parent)))
sys.path.insert(0, str(PYTHON_ROOT))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training_scope" / "LAM"))

# Compatibility aliases required by imports in the external training code.
_INTERNAL = ["V2", "V3", "V4", "V5", "V6", "V6_1", "V7"]
for _n in _INTERNAL:
    sys.modules.setdefault(f"LAM_{_n}", importlib.import_module(_n))
    sys.modules.setdefault(f"LAM_{_n}.tools", importlib.import_module(f"{_n}.tools"))

# Disable OpenCV's internal threads because the outer data thread pool already
# provides concurrency. Nested OpenCV mask-resize threads can race under
# multi-node load and cause a segmentation fault.
try:
    import cv2 as _cv2

    _cv2.setNumThreads(0)
except Exception:
    pass
decode_rows_from_shard = importlib.import_module(
    "Scale.common.shard_io"
).decode_rows_from_shard
_framedata = importlib.import_module("LAM_V3.tools._lam_v3_data")
_looplib = importlib.import_module("LAM_V5.tools.train_lam_v5")
_lambuilder = importlib.import_module("LAM_V2.tools.train_lam_action_readout")
_install_mask_patch = importlib.import_module(
    "LAM_V7.tools.train_v7_cached"
).install_v6_1_mask_patch

EVAL_ENTRY = Path(__file__).with_name("stage1_eval.py").resolve()


class _SubprocessRouter:
    """Route the external trainer's legacy evaluator to the release adapter."""

    def __getattr__(self, name):
        return getattr(subprocess, name)

    @staticmethod
    def run(command, *args, **kwargs):
        routed = list(command)
        if len(routed) > 1 and Path(str(routed[1])).name == "eval_lam_v2_g1.py":
            routed[1] = str(EVAL_ENTRY)
            print(f"[LAM masked training] routed evaluator={EVAL_ENTRY}", flush=True)
        return subprocess.run(routed, *args, **kwargs)


try:
    _LAM_400K = str(Path(os.environ["LAM_400K_LOCAL"]).expanduser().resolve())
except KeyError as exc:
    raise RuntimeError("LAM_400K_LOCAL is required") from exc
if not Path(_LAM_400K).is_file():
    raise FileNotFoundError(f"base LAM checkpoint is missing: {_LAM_400K}")


def _build_lam_pure_400k(name="LAM", device="cuda"):
    """Load the unmodified LAM_400k base without experimental delta checkpoints."""
    import time as _t

    _lambuilder._patch_sdpa()
    from external.lam.model import LAM

    _t0 = _t.time()
    print(f"[CD-LAM Stage 1] loading base LAM from {_LAM_400K} ...", flush=True)
    m = LAM(
        image_channels=3,
        lam_model_dim=1024,
        lam_latent_dim=32,
        lam_patch_size=16,
        lam_enc_blocks=24,
        lam_dec_blocks=24,
        lam_num_heads=16,
        ckpt_path=_LAM_400K,
    )
    m = m.to(device).eval()
    print(f"[CD-LAM Stage 1] base LAM ready in {_t.time() - _t0:.1f}s", flush=True)
    return m


def _patch_frame_source_to_shard():
    """Read embedded shard bytes while retaining the reference crop/scale path."""
    _framedata.decode_rows_parallel = decode_rows_from_shard
    if hasattr(_looplib, "decode_rows_parallel"):
        _looplib.decode_rows_parallel = decode_rows_from_shard


def _patch_base_lam_to_pure_400k():
    """Use the available LAM_400k base instead of unresolved delta variants."""
    _lambuilder.build_lam = _build_lam_pure_400k
    if hasattr(_looplib, "build_lam"):
        _looplib.build_lam = _build_lam_pure_400k


def main() -> int:
    _patch_frame_source_to_shard()
    _patch_base_lam_to_pure_400k()
    _install_mask_patch()  # Install frame_to_mask_idx-aware mask loading.
    _looplib.subprocess = _SubprocessRouter()
    print(
        "[CD-LAM Stage 1] frame source=embedded shard bytes; "
        f"base={_LAM_400K}; mask supervision enabled; starting.",
        flush=True,
    )
    return _looplib.main()


if __name__ == "__main__":
    sys.exit(main())
