# Installation and usage

## Install

```bash
git clone https://github.com/yufanwei/CD-LAM.git
cd CD-LAM
bash scripts/bootstrap.sh
```

For a manual installation, create a virtual environment and install
`-e ".[test]"`. All example paths are relative to the repository. Copy
`configs/paths.example.env` to `configs/paths.local.env` for your own storage
layout; do not commit it.

`scripts/run.sh` automatically sources `configs/paths.local.env` when present.
Set `CDLAM_PATHS_ENV=/absolute/path/to/profile.env` to select another profile.
The exported `CDLAM_*` asset paths override matching YAML `paths` fields and
are included in the effective configuration digest. When invoking the
installed `cdlam` command directly, source the profile first.

## Validate the source release

```bash
make check
```

The core smoke validates bridge normalization and serialization, the three
Stage-1 objective primitives, and a small FDCE computation. The training smoke
then performs real backward and optimizer steps for Stage 1, Stage 2, bridge
training, and Stage 3, writes checkpoints, and validates stage lineage:

```bash
bash scripts/run.sh train-smoke --output-root outputs/train-smoke --steps 2
```

Both use synthetic tensors and tiny CPU models. They validate code paths, not
paper-model convergence.

## Validate an asset-backed configuration

```bash
bash scripts/run.sh doctor \
  --strict \
  --config configs/pipeline_100h_2b.yaml
```

For 14B:

```bash
bash scripts/run.sh doctor \
  --strict \
  --config configs/pipeline_100h_14b.yaml
```

The configs encode the paper's main budgets: Stage 1 = 1,000 updates, Stage 2
= 2,000 updates, Stage 3 = 3,000 updates at 2B or 6,000 at 14B. Required
dataset/checkpoint fields are intentionally null in Git; fill them only in an
untracked local copy.

Inspect each typed production plan before launching an external adapter:

```bash
bash scripts/run.sh stage1 \
  --config configs/pipeline_100h_2b.yaml --dry-run --json
bash scripts/run.sh stage2 \
  --config configs/pipeline_100h_2b.yaml --dry-run --json
bash scripts/run.sh bridge-train \
  --config configs/pipeline_100h_2b.yaml --dry-run --json
bash scripts/run.sh stage3 \
  --config configs/pipeline_100h_2b.yaml --dry-run --json
```

A plan with missing assets, adapter, working directory, or incompatible action
contract returns exit code 2. External integration and resume requirements are
documented in [`TRAINING.md`](TRAINING.md).

## Optional mask and FDCE dependencies

Core smoke tests do not install SAM3 or CoWTracker.

- Configure SAM3 to generate the foreground masks used by the paper's Stage-1
  weighting and FDCE protocol.
- Configure CoWTracker only when computing FDCE.

Exact FDCE also requires the paper's seeding, erosion, visibility filtering,
resolution, and symmetric aggregation settings. See
[`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) and the upstream-license notes in
[`third_party/README.md`](../third_party/README.md).

Dataset splitting, transition alignment, Stage-1 mask fields, Stage-2 windows,
and the exact stride-4 22D bridge input are documented in
[`DATA.md`](DATA.md). The schemas are adapter contracts; the public core does
not claim to parse or train from them directly.

## Release limitations

The trained CD-LAM and ACWM checkpoints are pending upload. The runner does not
download or invent them. A source-only checkout supports full CPU integration
validation and production-plan checking; real 2B/14B execution still requires
a compatible external model adapter, assets, data, and GPU environment.
