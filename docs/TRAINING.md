# Training and adapters

CD-LAM separates method logic from model-specific infrastructure. The public
package owns configuration, stage ordering, action semantics, checkpoint
lineage, and validation. A production adapter owns the concrete LAM/ACWM model,
distributed launcher, dataset objects, and model checkpoint format.

## Execution modes

| mode | purpose | evidence it provides |
|---|---|---|
| `--dry-run` | resolve a typed plan without writing files | paths, assets, adapter, steps, device, action contract, blockers |
| `--synthetic` | run deterministic tiny PyTorch models on CPU | backward, optimizer, checkpoint, resume, and stage-lineage integration |
| external adapter | run a configured real model stack | adapter-defined GPU training and artifact evidence |

Synthetic mode is an integration test. It does not estimate paper quality or
memory requirements.

## Commands

Install the package, then use one config for all stages:

```bash
cdlam stage1       --config configs/pipeline_100h_2b.yaml --dry-run
cdlam stage2       --config configs/pipeline_100h_2b.yaml --dry-run
cdlam bridge-train --config configs/pipeline_100h_2b.yaml --dry-run
cdlam stage3       --config configs/pipeline_100h_2b.yaml --dry-run
```

A blocked dry run returns exit code 2. This is intentional: CI and launch
automation must not treat a plan with missing assets or incompatible action
coordinates as runnable.

The complete CPU training smoke is one command:

```bash
cdlam train-smoke --output-root outputs/train-smoke --steps 2 --json
```

Individual synthetic stages are also available:

```bash
cdlam stage1 --config configs/pipeline_100h_2b.yaml \
  --synthetic --device cpu --steps 2
```

## Pipeline configuration

Official templates live in `configs/`. Copy one to an ignored local file and
populate it; do not commit local storage paths or credentials.

Important fields:

| section | contract |
|---|---|
| `paths.project_root` | anchor for every relative path; independent of launch CWD |
| `paths.*manifest` | prepared Stage-1/2 or robot-action data index |
| `paths.base_acwm`, `paths.lam_init` | initial model assets |
| `paths.stage1_lam` | exact Stage-1 output consumed by Stage 2 and bridge training |
| `paths.stage2_acwm`, `paths.bridge_bundle` | exact parents consumed by Stage 3 |
| `adapters.*` | `python.module:factory_or_instance` integration reference |
| `optimizer_steps` | total optimizer-update target, not additional steps |
| `observed_checkpoint_steps` | optional audited step count for a selected partial artifact |
| `resume` | checkpoint and its completed-step declaration |
| `action_transform_id`, `source_stride` | semantic bridge/Stage-3 compatibility key |
| `stage3.working_directory` | explicit external resource root when an integration requires one |

Stage 2 requires training scope `D` in the production config. Bridge training
and Stage 3 must declare identical action-transform IDs and source strides.
Stage 3 also requires an explicit working directory for an external adapter.

## Production adapter interface

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

1. Run `make check`.
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
