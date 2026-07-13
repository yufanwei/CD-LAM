# Offline GPU-node setup

The supported newcomer installation downloads pinned source and Python
artifacts. When the training node cannot reach GitHub, package indexes, or model
hosts, prepare the complete checkout on an equivalent online GPU node and then
transfer it.

This procedure still uses one CD-LAM environment, `.venv`. SAM3 and CoWTracker
remain optional external source/cache tools; they do not create additional
CD-LAM environments.

## Compatibility requirement

The online staging node and offline target must match the supported runtime
contract:

- Linux x86-64 with glibc 2.35 or newer;
- CPython 3.10 available at a compatible host path;
- Ampere or Hopper NVIDIA GPU;
- NVIDIA driver 570.124.06 or newer; and
- the same absolute checkout destination when transferring `.venv`.

Virtual-environment entry-point scripts contain absolute interpreter paths.
Moving a prepared `.venv` to a different checkout path is unsupported. If the
target path must differ, prepare the checkout at that final absolute path on the
online node before transfer.

## Prepare on the online GPU node

Clone the source directly into the final checkout path, review the upstream
license, and run the normal GPU bootstrap:

```bash
git clone https://github.com/yufanwei/CD-LAM.git /srv/cdlam/CD-LAM
cd /srv/cdlam/CD-LAM
bash setup.sh --accept-base-license --with-models
```

This creates and validates:

- `.venv` with CPython 3.10, PyTorch `2.7.0+cu128`, CUDA 12.8 wheels, and all
  CD-LAM training/data/download/test tools;
- `.deps/acwm-runtime` from the pinned upstream commit plus verified overlay;
- the package download cache; and
- `artifacts/` containing the three hash-verified public CD-LAM entries.

Obtain the separately licensed NVIDIA base checkpoint, Cosmos video tokenizer,
Cosmos text encoder, and required datasets from their publishers. Keep Hugging
Face snapshots in their normal `refs`, `snapshots`, `blobs`, and relative-link
layout. Populate the ignored `configs/runtime.json` only with machine-local
paths; never place credentials in that file.

Before transfer, run the asset-aware preflight while the caches are complete:

```bash
bash run.sh runtime-doctor --stage all
bash run.sh pipeline --dry-run
```

The dry run may create disposable resolved configuration under the selected
output root. Remove that output before starting the real pipeline if needed.

## Transfer without rewriting the layout

Transfer the complete checkout while preserving permissions, symlinks, sparse
files, and timestamps. For example, from a machine that can reach the target:

```bash
rsync -aH --info=progress2 /srv/cdlam/CD-LAM/ \
  gpu-node:/srv/cdlam/CD-LAM/
```

For separately mounted model or dataset roots, transfer those roots to the exact
paths recorded in `configs/runtime.json`. Verify model publisher hashes and the
CD-LAM asset manifest after transfer. Do not replace Hugging Face snapshot
symlinks with flattened copies.

## Validate on the offline GPU node

Do not rerun the network bootstrap. Validate the transferred environment,
source tree, live driver, CUDA runtime, and an actual optimizer update:

```bash
cd /srv/cdlam/CD-LAM
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1

.venv/bin/python scripts/model_runtime_doctor.py --check-driver --gpu 0
.venv/bin/python scripts/gpu_smoke.py --gpu 0
bash run.sh runtime-doctor --stage all
bash run.sh pipeline --dry-run
```

Real stage subprocesses set `HF_HUB_OFFLINE=1` and skip default checkpoint
downloads. A missing requested revision, tokenizer, text encoder, base
checkpoint, or dataset therefore fails locally instead of silently contacting a
model host.

## Package-cache boundary

A standalone package-only GPU capsule is not published in this release. Such a
capsule would need every artifact from the pinned upstream `uv.lock`, the exact
Git source bundle, CD-LAM supplements, direct VCS dependencies, and a manifest
bound to Python, operating system, architecture, glibc, CUDA wheel line, and
source hashes. Transferring a fully validated same-path checkout is the
supported offline-node procedure.

## Optional metric assets

SAM3 is needed only to generate new Stage-1/FDCE masks. CoWTracker is needed
only to generate new FDCE tracks. Reuse compatible precomputed masks and tracks
when available. Otherwise, review each upstream license on the online node,
stage its pinned source and weights, and transfer those caches separately. Their
source and weights are not covered by the CD-LAM repository license.
