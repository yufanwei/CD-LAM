# Offline and fresh-machine setup

The normal quick start creates an isolated virtual environment and may use the
Python package index. A machine with no network access needs two transferred
items:

1. the CD-LAM source directory; and
2. a platform-specific core runtime cache created from a known-good isolated
   environment on a compatible machine.

The cache contains Python packages, one CD-LAM wheel, and the Ruff executable.
It is bound to the Python minor version, implementation tag, operating system,
and CPU architecture recorded in `offline-cache.json`. It is not a source
artifact and should not be committed to Git or uploaded to the model
repository.

## Build the transferable core cache

Run the cache builder with the Python environment whose package set should be
captured. The builder interpreter may be separate when the runtime environment
does not contain `setuptools>=77`:

```bash
/path/to/runtime/bin/python scripts/offline_cache.py create \
  --builder-python /path/to/build-tools/bin/python \
  --output /transfer/cdlam-core-cache
```

The source runtime must contain the exact distributions in
`requirements.lock`. The builder must contain `build`, `wheel`, and
`setuptools>=77`. Creation is atomic and refuses to overwrite an existing
target. The manifest hashes the complete package tree, wheel, and executable.

## Install with no package-index access

On the target machine, start from the transferred source directory:

```bash
export CDLAM_VENV=/tmp/cdlam-core
export PIP_NO_INDEX=1
bash scripts/bootstrap.sh --offline-cache /transfer/cdlam-core-cache
```

This creates a venv with `include-system-site-packages = false`, validates the
cache before copying it, installs the prebuilt wheel with `--no-index
--no-deps`, runs `pip check`, and executes the release, test, data, and
optimizer-smoke gates. The command rejects `--with-models` and
`--with-metrics` in offline mode because those options are network fetches.

`--reuse-system-runtime` is available only as an explicit integration escape
hatch for an already validated CUDA environment. It is not clean-room
evidence and is mutually exclusive with `--offline-cache`.

## Heavy model and metric assets

The core cache deliberately excludes base models, CD-LAM checkpoints,
datasets, SAM3, and CoWTracker. Stage these separately under a machine-local
root and verify their revisions and hashes before creating the runtime profile.
Keep three environments when CUDA/vision dependencies conflict:

| environment | contents |
|---|---|
| core | package, tests, configs, bridge and metric reduction |
| model runtime | ACWM/LAM source, Cosmos caches, CUDA training dependencies |
| metric runtime | SAM3 masks and CoWTracker tracks |

Hugging Face cache snapshots must retain their `refs`, `snapshots`, `blobs`,
and relative links. Set `HF_HOME` to the transferred cache root and set
`HF_HUB_OFFLINE=1` only after the requested revisions are present. SAM3 and
CoWTracker remain external because their licenses and large weights are not
the same as the CD-LAM source license.

The upstream base revision alone does not contain every real Stage
1/2/bridge/Stage 3 entry. The required source-only overlay is bundled at
`third_party/acwm_overlay`; stage it from a transferred pinned base checkout
with `internal/tools/stage_acwm_runtime.py`. The command verifies the base
commit, modified-base hashes, every overlay hash, required runtime paths, and
the resulting provenance. Runtime checks fail when any declared checkpoint or
source contract is absent; they never fall back to similarly named workspace
files.
