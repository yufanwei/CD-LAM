# Reproducible 2B model environment

This page covers only the isolated CUDA environment used by the real 2B
training launchers. It does not install the core/data environment, download a
model, accept a model license, prepare a dataset, reserve a GPU, or launch
training.

## Supported host contract

The virtual-environment path targets the pinned upstream CUDA 12.8 / PyTorch
2.7 dependency graph on Linux x86-64:

- CPython 3.10 and glibc 2.35 or newer;
- an Ampere or Hopper NVIDIA GPU;
- NVIDIA driver 570.124.06 or newer;
- `git`, the Python `venv` module, and `ffmpeg` available on `PATH`;
- approximately 30 GB free for the isolated packages and download cache.

Large CUDA wheels can exceed a short HTTP timeout. The bootstrap uses 300
seconds per request by default; set `CDLAM_MODEL_HTTP_TIMEOUT` to a larger
positive integer on a slow link. Put `CDLAM_MODEL_UV_CACHE` on local storage
when the checkout resides on a network filesystem.

The linked CD-LAM execution check used Ubuntu 22.04.5, CPython 3.10.18, an H100
80 GB, driver 580.126.09, and `torch==2.7.0+cu128`. It completed one real 2B
optimizer update per training stage. That is execution and checkpoint-lineage
evidence, not convergence or paper-result evidence.

The pinned upstream Dockerfile references
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`. CD-LAM did not use that image for
the recorded Ubuntu 22.04 H100 run, so the image is a reproducible upstream base
reference rather than a CD-LAM container-validation claim. Blackwell virtual
environment support is not claimed; the pinned upstream documentation requires
its separate Docker path for Blackwell.

The setup command does not install or upgrade the host NVIDIA driver, CUDA
kernel driver, container runtime, operating-system packages, or GPU firmware.
Those remain administrator-owned host prerequisites.

## One-command setup

Start from a fresh CD-LAM source checkout on a compatible host. Review the
pinned upstream source license, then run:

```bash
CDLAM_ACCEPT_BASE_LICENSE=yes bash scripts/bootstrap_model_runtime.sh
```

The command performs four fail-closed operations:

1. fetches the exact upstream source revision, applies the manifest-verified
   61-file CD-LAM overlay, and binds the complete 737-file base-plus-overlay tree
   under `.deps/acwm-runtime`;
2. verifies that complete runtime-tree digest, including unmodified upstream
   files, plus the source revision, overlay manifest, and SHA-256 of the upstream
   `uv.lock`;
3. bootstraps `uv==0.9.7` in a disposable temporary environment and runs the
   upstream lock with `--no-dev --extra cu128` into `.deps/model-env`;
4. installs the exact Lightning runtime supplements and PyTorch3D source
   revision omitted by the upstream project metadata, deterministically places
   the locked headless OpenCV payload last, then runs the model-environment
   doctor.

The staged LAM implementation imports Lightning, but the pinned upstream
project graph does not declare it. CD-LAM therefore locks `lightning`,
`pytorch-lightning`, `lightning-utilities`, and `torchmetrics` to the versions
used by the validated runtime. They are installed without changing the
upstream-resolved CUDA graph, and the doctor checks both their metadata and a
real `lightning` import.

CD-LAM uses only `pytorch3d.transforms` for rotation conversion. Disabling the
optional PyTorch3D C++/CUDA operators avoids an undeclared compiler/toolkit
dependency while preserving the imported surface used by the runtime.

The upstream graph contains both `opencv-python` and
`opencv-python-headless==4.11.0.86`; those distributions own overlapping `cv2`
paths. The bootstrap retains both locked metadata records but reinstalls that
exact headless wheel last. This makes the real `cv2` import independent of an
undeclared host `libGL.so.1` package without weakening dependency validation.

The target is never merged with `.venv` or the active shell environment. If
`.deps/model-env` already exists, the command validates it and makes no changes.
Use a different target instead of overwriting an environment:

```bash
CDLAM_ACCEPT_BASE_LICENSE=yes \
  bash scripts/bootstrap_model_runtime.sh \
    --python /absolute/path/to/python3.10 \
    --environment /absolute/path/to/new-model-env
```

Inspect every operation without creating a source checkout or environment:

```bash
bash scripts/bootstrap_model_runtime.sh --dry-run
```

The transitive package graph is not duplicated in a hand-maintained
requirements file. `configs/model_runtime.lock.json` binds the source commit,
upstream lock hash, installer version, CUDA extra, platform contract, critical
package versions, and external asset revisions. The upstream `uv.lock` contains
the complete artifact URLs and hashes.

That pinned upstream graph has exactly two metadata inconsistencies: its
`megatron-core==0.14.0` metadata declares `numpy<2`, while the upstream uv
override selects `numpy==2.2.6`; and the `cosmos-predict2==1.4.1` workspace
package still declares `cosmos-oss==0.1.0`, while the same workspace builds
`cosmos-oss==1.4.1`. Both exact conflicts are recorded structurally in the
CD-LAM lock. The doctor evaluates every installed `Requires-Dist` entry with
`importlib.metadata` and `packaging`; it passes only when the observed conflict
set equals that two-row allowlist. Any new conflict, resolved expected conflict,
missing dependency, duplicate distribution, or invalid metadata fails setup.

## Environment doctor

The default doctor is CPU-safe: it recomputes the full 737-file runtime-tree
digest, then checks source provenance, isolation, executables, exact critical
distribution versions, the complete dependency-metadata conflict set, real
isolated imports of every required module, and editable package origins without
loading a checkpoint or touching CUDA.

```bash
python scripts/model_runtime_doctor.py
```

On the GPU host, check driver visibility and PyTorch CUDA compatibility without
launching a training kernel:

```bash
python scripts/model_runtime_doctor.py --check-driver --gpu 0
```

The second command queries `nvidia-smi` and PyTorch device availability. It is
not a model load, memory allocation test, optimizer update, or GPU acceptance
run.

Point the ignored real-runtime profile at the isolated executables:

```json
{
  "paths": {
    "python": ".deps/model-env/bin/python",
    "torchrun": ".deps/model-env/bin/torchrun"
  }
}
```

After every model and data path is populated, use the existing asset-aware
doctor and dry run:

```bash
bash scripts/run.sh runtime-doctor --stage all
bash scripts/run.sh pipeline --dry-run
```

## Assets deliberately excluded

Environment setup never downloads these separately licensed or user-owned
inputs:

- the compatible NVIDIA base LAM and 2B world-model checkpoint;
- the Cosmos-Predict2.5 video tokenizer;
- the Cosmos-Reason1 text encoder;
- CD-LAM checkpoints, bridges, datasets, masks, tracks, or evaluation media;
- SAM3 or CoWTracker.

Their exact repositories and immutable revisions are recorded in
`configs/model_runtime.lock.json` and `third_party/dependencies.lock.json`.
Acquire gated assets directly from their publishers, retain the Hugging Face
snapshot layout, and keep the model host offline only after every requested
revision is present locally.
