"""Stage-2 masked world-model entry point, executed once per torchrun rank.

The entry point fine-tunes the configured 2B world model on LAM latent
conditions. The declared training scope covers the final four DiT blocks and
the action embedder, while conditioning dropout encourages use of the latent
condition. The external implementation identifier
``lamwm_pipline.tools.train_wm_compat_real`` is retained for compatibility.

Before calling the external trainer, this module performs three integration
tasks:

1. Decode a 13-frame window from bytes embedded in an 18-column shard. The
   manifest stores each location as ``<shard_path>::<row_index>`` and the
   patched decoder preserves the reference crop/scale path.
2. Build the LAM encoder through the external registry and remap checkpoint
   references to a configured local model root when available.
3. Call the manifest-capable external trainer directly so encoded shard
   references are not treated as standalone MP4 paths.

Input records come from ``wm_manifest.parquet``, materialized by
``build_wm_manifest.py`` with the encoded shard location in ``video_path``.
"""

import importlib
import os
import sys
from pathlib import Path

try:
    REPO = Path(os.environ["CDLAM_ACWM_ROOT"]).expanduser().resolve()
except KeyError as exc:
    raise RuntimeError("CDLAM_ACWM_ROOT is required") from exc
PYTHON_ROOT = Path(os.environ.get("CDLAM_PYTHON_ROOT", str(REPO.parent)))
REGISTRY_DIR = (
    Path(
        os.environ.get(
            "CDLAM_LAM_REGISTRY_DIR",
            str(REPO / "lamwm_pipline" / "configs"),
        )
    )
    .expanduser()
    .resolve()
)
sys.path.insert(0, str(PYTHON_ROOT))
sys.path.insert(0, str(REPO))
sys.path.insert(
    0, str(REPO / "finetune_4-30" / "scripts")
)  # Supports lam_loader's top-level `from model import ...`.

decode_window_from_shard = importlib.import_module(
    "Scale.common.shard_io"
).decode_window_from_shard


def _patch_decode_window_to_shard():
    """Replace the manifest decoder with the embedded-shard implementation.

    The external trainer calls its module-level ``_decode_window`` during both
    probing and sampling. Encoded video references use
    ``<shard_path>::<row_index>``.
    """
    import lamwm_pipline.tools.train_wm_compat_real as T

    def _decode_window_shard(video, start_frame, stop_frame, h, w):
        sp, ri = str(video).rsplit("::", 1)
        return decode_window_from_shard(
            sp, int(ri), int(start_frame), int(stop_frame), wm_hw=(int(h), int(w))
        )

    T._decode_window = _decode_window_shard


def _patch_lam_checkpoint_validation():
    """Fail before model construction when a registry checkpoint is missing."""
    import lamwm_pipline.src.lam_loader as L

    _orig = L.get_lam_entry

    def _patched(lam_id):
        e = dict(_orig(lam_id))
        checkpoint = Path(str(e.get("ckpt_path", ""))).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"LAM registry entry {lam_id!r} points to a missing checkpoint: {checkpoint}"
            )
        e["ckpt_path"] = str(checkpoint)
        return e

    L.get_lam_entry = _patched


def _patch_registry_config():
    """Point every rank at the profile-selected registry directory."""
    import lamwm_pipline.src.registry as R

    R.CONFIG_DIR = REGISTRY_DIR


def main() -> int:
    _patch_registry_config()
    _patch_decode_window_to_shard()
    _patch_lam_checkpoint_validation()
    import lamwm_pipline.tools.train_wm_compat_real as T

    print(
        "[CD-LAM Stage 2] frame source=embedded shard bytes; "
        "LAM registry validation enabled; manifest training starting.",
        flush=True,
    )
    # The external main function parses sys.argv; this entry point forwards the same arguments.
    T.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
