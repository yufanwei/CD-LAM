# Installation and usage

## Install and validate the source

```bash
git clone https://github.com/yufanwei/CD-LAM.git
cd CD-LAM
bash scripts/bootstrap.sh
make check
```

The bootstrap creates `.venv` for the source checks, data tools, and lightweight
CPU integration tests. The complete synthetic training check performs backward
and optimizer steps for Stage 1, Stage 2, bridge training, and Stage 3, then
verifies checkpoint lineage:

```bash
bash scripts/run.sh train-smoke --output-root outputs/train-smoke --steps 2
```

Synthetic tests validate the release plumbing; they do not estimate model
quality, GPU memory, or paper-model convergence.

## Prepare the real 2B runtime

Keep the core/data environment and the CUDA model environment separate. The
runtime profile names the Python and `torchrun` executables from the model
environment, so training does not depend on whichever shell environment happens
to be active.

After reviewing the upstream terms, create the pinned isolated model
environment and stage the manifest-checked NVIDIA source plus CD-LAM overlay.
Then verify both the environment and the live CUDA driver:

```bash
CDLAM_ACCEPT_BASE_LICENSE=yes bash scripts/bootstrap_model_runtime.sh
python scripts/model_runtime_doctor.py --check-driver
cp configs/runtime.example.json configs/runtime.json
```

Obtain the required base checkpoint and released CD-LAM assets separately, then
set their local paths and the prepared-data paths in the ignored
`configs/runtime.json`. No credential belongs in that file. Validate every
stage and print the exact launch commands before using a GPU:

```bash
bash scripts/run.sh runtime-doctor --stage all
bash scripts/run.sh pipeline --dry-run
```

The doctor fails closed for missing source files, model files, tokenizer or text
encoder cache entries, data manifests, action metadata, incompatible bridge
lineage, or incomplete checkpoints. The dry run writes resolved configuration
under `paths.output_root`; remove that disposable output root or select a fresh
one before the real launch.

## Train

The real 2B commands use the pinned implementation staged at
`.deps/acwm-runtime`:

```bash
# Run the complete linked chain.
bash scripts/run.sh pipeline --allow-gpu

# Or run one stage with the parent assets declared in runtime.json.
bash scripts/run.sh stage1 --allow-gpu
bash scripts/run.sh bridge --allow-gpu
bash scripts/run.sh stage2 --allow-gpu
bash scripts/run.sh stage3 --allow-gpu
```

In the complete pipeline, the newly written Stage-1 checkpoint feeds both the
bridge and Stage 2. The newly written bridge and Stage-2 checkpoint then feed
Stage 3. Any failed subprocess or validator stops the chain.

Stage 2 is the no-bridge path: the world model is conditioned directly by the
32D latent actions from the selected LAM. The bridge is needed only for Stage 3
or inference from recorded 22D robot actions. See [`TRAINING.md`](TRAINING.md)
for configuration, resume, custom-backbone, and acceptance contracts.

The portable YAML files remain useful for validating paper budgets or planning
a custom backbone integration:

```bash
bash scripts/run.sh doctor --strict --config configs/pipeline_100h_2b.yaml
bash scripts/run.sh plan-stage1 \
  --config configs/pipeline_100h_2b.yaml --dry-run --json
```

The bundled concrete runtime targets the released 2B implementation. The 14B
YAML records the manuscript protocol but requires a compatible user-supplied
backbone adapter and 14B assets; this release does not publish 14B weights.

## Data and optional metrics

The repository directly prepares official-schema AgiBot episodes and a bounded
official-format EgoDex subset. Dataset splitting, media decoding, transition
alignment, Stage-1 mask fields, Stage-2 windows, and the stride-four 22D bridge
input are documented in [`DATA.md`](DATA.md).

SAM3 and CoWTracker are not installed by the core bootstrap:

- SAM3 generates the masks required for paper-equivalent Stage-1 weighting and
  FDCE.
- CoWTracker is needed only to compute FDCE tracks.

Compatible precomputed masks and tracks can be used instead. Exact FDCE also
requires the documented seeding, erosion, visibility filtering, resolution,
and aggregation settings. See [`EVALUATION.md`](EVALUATION.md) for the runnable
scoring workflow, [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) for the complete
protocol, and the license notes in
[`third_party/README.md`](../third_party/README.md).

## Release limitations

The compact three-entry 2B snapshot has been staged and checked locally, but an
immutable Hugging Face revision is not published yet. Until it is, the model
downloader rejects the legacy `main` layout before transferring weights. Real
training also requires the separately licensed NVIDIA base checkpoint,
user-obtained datasets, compatible CUDA/PyTorch packages, and sufficient GPU
memory. The source release validates execution and lineage; it does not claim
to regenerate the manuscript tables from a fresh clone.
