# Training runtime

CD-LAM separates stable public commands from the pinned model implementation.
The release owns configuration, stage ordering, action semantics, checkpoint
lineage, validators, and the concrete 2B launch wrappers. The staged ACWM tree
is built from one pinned upstream commit plus the small manifest-checked overlay
under `third_party/acwm_overlay/`.

## Execution modes

| mode | purpose | evidence it provides |
|---|---|---|
| `runtime-doctor` | validate the complete real runtime without training | executables, source closure, local models/data, imports, action contract |
| `pipeline --dry-run` | print all real launch commands without GPU work | resolved paths, steps, devices, and subprocess routing |
| `--synthetic` | run deterministic tiny PyTorch models for source tests | backward, optimizer, checkpoint, resume, and stage-lineage integration |
| real stage or pipeline | run the pinned 2B implementation | real GPU updates, stage validators, and bound output lineage |

Synthetic mode is an integration test. It does not estimate paper quality or
memory requirements.

Install and validate the single locked GPU environment before a real stage; see
[GPU installation contract](MODEL_RUNTIME.md). The same `.venv` handles real
training, planning, data preparation, downloads, evaluation, and source checks.

## Commands

Create one untracked runtime profile for all real stages:

```bash
cp configs/runtime.example.json configs/runtime.json
bash run.sh runtime-doctor --stage all
bash run.sh pipeline --dry-run
```

Run the complete pipeline or an individual stage with:

```bash
bash run.sh stage1
bash run.sh bridge
bash run.sh stage2
bash run.sh stage3
bash run.sh pipeline
```

The full pipeline produces Stage 1 first, uses that exact checkpoint for both
bridge and Stage 2, binds the new bridge to the action contract, and then uses
the newly produced Stage-2 state and bridge for Stage 3. A failed stage stops
the chain. Independent stages require the parent checkpoints declared in the
profile.

The complete deterministic source-integration smoke is one command:

```bash
.venv/bin/cdlam train-smoke --output-root outputs/train-smoke --steps 2 --json
```

Custom planner routes remain available without launching a real model:

```bash
bash run.sh plan-stage1 \
  --config configs/pipeline_100h_2b.yaml --dry-run --json
```

## Pipeline configuration

`configs/runtime.example.json` is the real 2B template. Relative paths are
resolved from `workspace`, itself resolved beside the profile. Do not commit a
populated profile, storage path, or credential.

Important fields:

| section | contract |
|---|---|
| `workspace` | anchor for every relative path; independent of launch CWD |
| `paths.python`, `paths.torchrun` | `.venv` executables created by the GPU bootstrap |
| `paths.acwm_root` | staged, manifest-verified runtime, normally `.deps/acwm-runtime` |
| data paths | Stage-1 indices, bridge cache, Stage-2 manifest, and Stage-3 dataset YAML |
| base paths | exact LAM checkpoint, 2B distributed checkpoint, and local HF cache |
| independent-stage paths | released or user-supplied Stage-1, bridge, and Stage-2 parents |
| `stage1`/`bridge`/`stage2`/`stage3` | optimizer budgets, batch sizes, cadence, and evaluation settings |
| `launch` | visible GPU list, process count, and seed |

Stage 2 and Stage 3 use the release-pinned scope and action dimensions. The
doctor rejects missing data, base files, tokenizer/text-encoder cache entries,
source adapters, public experiment aliases, or action-contract metadata before
launch.

## Custom backbone adapter interface

An adapter subclasses `cd_lam.adapters.StageAdapter`:

```python
from cd_lam.adapters import StageAdapter
from cd_lam.config import StageName


class MyStageAdapter(StageAdapter):
    @property
    def identity(self):
        return "my_project.cd_lam_adapter"

    @property
    def supported_stages(self):
        return frozenset({StageName.STAGE1})

    def validate(self, context):
        # Validate source revision, checkpoint schema, data manifest,
        # trainable scope, device topology, and output safety here.
        ...

    def run(self, context):
        # Perform training and return cd_lam.training.common.StageResult.
        ...
```

Configure it with:

```yaml
adapters:
  stage1: my_project.adapters:MyStageAdapter
```

`validate()` runs before the adapter may mutate outputs or start distributed
workers. `run()` must produce the checkpoint declared by the plan and return a
passing result with finite losses, the exact config digest, total step count,
seed, and stable adapter identity. The public runner rejects inconsistent
metadata or a missing checkpoint.

Production adapters should additionally record:

- source commit and dirty-patch digest;
- data-manifest digest and episode split;
- complete trainable-parameter audit;
- parent checkpoint path, role, and SHA-256;
- optimizer and scheduler state for true resume;
- world size, per-rank batch size, and effective global batch size;
- action layout, units, transform ID, stride, and statistics hashes;
- acceptance verdict and failure reason.

## Stage contracts

### Stage 1

Stage 1 fine-tunes a 32D latent action model with embodiment-centric
reconstruction, action-centric contrast, and latent-space calibration. The
paper target is 1,000 optimizer updates. A historical 150-update checkpoint is
a partial artifact and must not occupy the paper-complete role.

Required validation includes foreground-mask alignment, primitive-label
coverage, identity-transition sampling, finite loss components, and optimizer
resume.

### Stage 2

Stage 2 consumes the exact Stage-1 LAM output directly; it does not use the
robot-action bridge. The paper target is 2,000 optimizer updates. Source FPS
must come from validated manifest metadata, and the full declared scope must be
audited before and after one update.

### Bridge training

Bridge training pairs a raw 22D adjacent robot-action delta with a 32D target
from the exact Stage-1 LAM. Split by episode before fitting statistics. The
checkpoint must contain the MLP and both action-side and latent-side
normalization statistics.

### Stage 3

Stage 3 consumes the exact Stage-2 state and bridge. For the AgiBot Alpha
contract, the dataset loader's normalized block-anchor cumulative deltas are
converted to raw adjacent stride-four deltas before bridge standardization.
The conversion must happen before reduced-precision casting.

The paper targets are 3,000 optimizer updates for 2B and 6,000 for 14B. A
production run also needs own-condition, zero-condition, and shuffled-condition
checks to demonstrate that the model responds to the intended action signal.

## Resume and lineage

`--steps` always means the desired total. A checkpoint at step 1 resumed with
`--steps 3` performs two additional optimizer updates. Synthetic checkpoints
validate stage, config digest, seed, adapter identity, and completed steps
before loading model and optimizer state.

The ordered smoke runner hashes actual outputs:

```text
Stage-1 hash -> Stage-2 metadata
Stage-1 hash -> bridge metadata
Stage-2 hash + bridge hash -> Stage-3 metadata
```

A production adapter should follow the same role-based chaining. Static
fallback paths are not acceptable evidence that a newly produced Stage-1
checkpoint was used downstream.

## Before a large run

1. Run `bash run.sh check`.
2. Run all four production dry runs and resolve every blocker.
3. Decode one golden sample per stage and validate FPS, timestamps, masks,
   frame order, and action alignment.
4. Complete one real optimizer update per stage.
5. Load each checkpoint and resume for one more update.
6. Verify trainable-parameter and gradient audits.
7. Verify bridge-training and Stage-3 inputs match numerically.
8. Require a nonzero process exit for failed acceptance criteria.

The complete promotion criteria are in
[Training correctness and release gates](TRAINING_CORRECTNESS.md).
