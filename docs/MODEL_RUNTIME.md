# GPU installation contract

CD-LAM has one supported repository environment: `.venv`. The same environment
provides the real 2B stage launchers, data and model download tools, evaluation
utilities, and source-validation commands.

## Supported host

The pinned runtime targets:

- Linux x86-64 with glibc 2.35 or newer;
- CPython 3.10;
- PyTorch `2.7.0+cu128` with CUDA 12.8 wheels;
- an Ampere or Hopper NVIDIA GPU;
- NVIDIA driver 570.124.06 or newer;
- `git`, the Python `venv` module, and `ffmpeg` on `PATH`; and
- approximately 30 GB free for packages and the download cache.

The CUDA runtime is supplied by the pinned wheels, so a separate system CUDA
toolkit is not required. The setup command does not install or upgrade the host
NVIDIA driver, operating-system packages, container runtime, or GPU firmware.

The linked CD-LAM execution check used Ubuntu 22.04.5, CPython 3.10.18, an H100
80 GB, driver 580.126.09, and `torch==2.7.0+cu128`. It completed one real 2B
optimizer update per training stage. That is execution and checkpoint-lineage
evidence, not convergence or paper-result evidence.

The pinned upstream Dockerfile references
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`. CD-LAM did not use that image for
the recorded Ubuntu 22.04 H100 run, so it is an upstream reference rather than
a CD-LAM container-validation claim. Blackwell support is not claimed by this
virtual-environment path.

## Install

Start from a fresh checkout, review the pinned upstream source license, and run:

```bash
bash setup.sh --accept-base-license
```

Use a different visible GPU for validation with:

```bash
bash setup.sh --accept-base-license --gpu 1
```

The bootstrap fails closed while performing these operations:

1. fetch the exact upstream source revision, apply the manifest-verified CD-LAM
   overlay, and bind the complete base-plus-overlay tree under
   `.deps/acwm-runtime`;
2. verify the source revision, complete runtime-tree digest, overlay manifest,
   and upstream `uv.lock` SHA-256;
3. run the upstream locked `cu128` dependency graph into `.venv` with CPython
   3.10;
4. install the locked HDF5, Lightning, PyTorch3D, test, data, build, and download
   tools into that same `.venv` without replacing the CUDA graph;
5. validate exact critical distributions, dependency metadata, isolated module
   imports, source origins, NVIDIA driver visibility, PyTorch CUDA 12.8, and the
   selected GPU; and
6. perform a real CUDA forward, backward, optimizer update, and synchronization,
   followed by the deterministic source, data, and package gates.

The bootstrap creates no fallback environment and never exposes packages from
the parent interpreter. Inspect the operations without creating files with:

```bash
bash setup.sh --accept-base-license --dry-run
```

Large CUDA wheels can exceed short HTTP timeouts. Set
`CDLAM_MODEL_HTTP_TIMEOUT` to a larger positive integer on a slow link. Put
`CDLAM_MODEL_UV_CACHE` on local storage when the checkout resides on a network
filesystem.

The transitive package graph is not duplicated in a hand-maintained generic
requirements file. `configs/model_runtime.lock.json` binds the source commit,
upstream lock hash, installer version, CUDA extra, platform contract, critical
package versions, and external asset revisions. The upstream `uv.lock` contains
the complete artifact URLs and hashes.

The pinned graph has two recorded upstream metadata inconsistencies:
`megatron-core==0.14.0` declares `numpy<2` while the upstream override selects
`numpy==2.2.6`, and `cosmos-predict2==1.4.1` declares `cosmos-oss==0.1.0` while
the workspace installs `cosmos-oss==1.4.1`. The environment doctor accepts only
that exact two-row allowlist. Any new conflict, missing dependency, duplicate
distribution, invalid metadata, or changed expected conflict fails setup.

## Validate again

Bootstrap already performs the live GPU validation. To repeat it on GPU 0:

```bash
.venv/bin/python scripts/model_runtime_doctor.py --check-driver --gpu 0
.venv/bin/python scripts/gpu_smoke.py --gpu 0
```

The first command checks the complete staged source, package graph, driver, and
CUDA visibility. The second performs an actual CUDA optimizer step; it is not a
model load or a paper-scale training run.

The real runtime profile uses the same environment:

```json
{
  "paths": {
    "python": ".venv/bin/python",
    "torchrun": ".venv/bin/torchrun"
  }
}
```

After populating every model and data path, validate and print the routed
commands before launch:

```bash
cp configs/runtime.example.json configs/runtime.json
bash run.sh runtime-doctor --stage all
bash run.sh pipeline --dry-run
```

## Assets not installed by default

The base bootstrap does not download:

- the compatible NVIDIA base LAM and 2B world-model checkpoint;
- the Cosmos-Predict2.5 video tokenizer;
- the Cosmos-Reason1 text encoder;
- CD-LAM checkpoints, bridges, datasets, masks, tracks, or evaluation media;
- SAM3 or CoWTracker source and weights.

Add `--with-models` to download and verify the three published CD-LAM entries.
Acquire the separately licensed NVIDIA and dataset assets from their publishers,
retain their immutable snapshot layouts, and fill only local paths in the
ignored runtime profile.

SAM3 and CoWTracker remain optional external source and cache tools. They are
needed only when generating new masks or tracks and do not create another
CD-LAM environment. See [Offline GPU-node setup](OFFLINE.md) before moving a
prepared installation to a model host without network access.
