# ACWM runtime overlay

This directory is the small, source-only CD-LAM integration overlay for the
pinned NVIDIA ACWM implementation. It contains 49 files totaling less than
1 MiB; it does not contain base-model weights, tokenizer weights, text-encoder
weights, datasets, caches, or generated outputs.

`manifest.json` binds every overlay byte to NVIDIA source commit
`02f119b759d5c7f84a399fdeea3c6e82e7ed6cff`. Modified upstream files also
record their original SHA-256, so staging fails if either the base or overlay
drifts. Build or verify the isolated runtime with:

```bash
CDLAM_ACCEPT_BASE_LICENSE=yes bash scripts/run.sh fetch-base
python internal/tools/stage_acwm_runtime.py \
  --verify-existing --output .deps/acwm-runtime
```

CD-LAM additions live under `cdlam_integration/`, grouped by latent-action,
bridge, world-model, and robot-action stages. Modified upstream Cosmos and
dataset files retain their upstream paths so the pinned package can load them
without runtime monkey-patching.

The overlay is Apache-2.0 source derived from and intended to be applied to the
Apache-2.0 upstream implementation. Upstream copyright and license notices
remain applicable. Model weights and other external assets have separate
terms; see [`../dependencies.lock.json`](../dependencies.lock.json) and
[`../README.md`](../README.md).
