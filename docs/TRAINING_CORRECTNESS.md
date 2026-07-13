# Training correctness and release gates

> **Status: paper-equivalent training is not yet accepted.** Host-side source
> gates pass, and the bundled 2B GPU runtime has tested working-directory and
> action-coordinate fixes. One linked H100 invocation completed a real
> optimizer update in Stage 1, Stage 2, and Stage 3 and trained a new bridge;
> Stage 2 used the newly written Stage-1 checkpoint, and Stage 3 used the newly
> written Stage-2 checkpoint plus that bridge. This establishes execution and
> smoke-scale lineage, not checkpoint resume, convergence, or a complete
> paper-budget lineage. Do not claim end-to-end paper reproduction until every
> acceptance gate in this document passes.

This document separates two kinds of evidence:

- **Paper protocol** is the normative experimental specification described in
  the manuscript and summarized in [`PIPELINE.md`](PIPELINE.md).
- **Observed artifacts** are the launchers, configs, checkpoints, logs, and
  loader behavior inspected during the release audit. An artifact can be
  useful for debugging without being paper-equivalent.

A process that starts, a checkpoint that loads, or a directory with a
plausible name is not sufficient evidence of training correctness. Correctness
requires verified data coordinates, checkpoint lineage, trainable scope,
optimizer-update counts, and stage-to-stage handoff.

## Paper protocol baseline

| role | paper protocol |
|---|---|
| `stage1_lam_final` | LAM debiased fine-tuning with all three CD-LAM objectives; 1,000 optimizer updates for the main result |
| `stage2_acwm_final` | ACWM debiased fine-tuning on latents from `stage1_lam_final`; 2,000 optimizer updates |
| `stage3_bridge` | map the documented 22D robot-action transition into the matching 32D Stage-1 latent space |
| `stage3_acwm_final_2b` | robot-action adaptation initialized from the matching Stage-2 state; 3,000 optimizer updates |
| `stage3_acwm_final_14b` | robot-action adaptation initialized from the matching Stage-2 state; 6,000 optimizer updates |

The 100-hour debiasing tier is the main paper setting. Update counts refer to
optimizer updates, not filenames, samples, epochs, or logging events.

## Blocking discrepancies

### B1. Stage-3 working-directory dependency

**Paper/public expectation.** A release command must resolve code, configs,
data, and checkpoints from explicit configuration. It must not depend on the
shell's current directory or an unrecorded checkout layout.

**Historical artifact state.** The pre-release Stage-3 path relied on being
launched from a particular upstream source checkout. Relative imports and
resource lookups could change with the shell working directory.

**Implemented mitigation.** The bundled wrapper resolves the upstream entry
by absolute path, validates its resource root, changes working directory only
for the external call, and restores the caller's directory. Host-side tests verify
that behavior, and the linked 2B smoke completed the real one-update Stage-3
path from the public wrapper. Checkpoint resume remains required acceptance
evidence.

**Why this blocks training.** A command may fail immediately, import a
different module, or consume a different config depending on where it is run.
Such a launch is not portable or auditable.

**Acceptance evidence.** All of the following must pass:

1. The same Stage-3 dry run resolves identical inputs when launched from the
   repository root and from an unrelated temporary working directory.
2. Every source, data, output, and checkpoint path appears in the resolved
   config/provenance record.
3. A one-optimizer-update Stage-3 run completes without an implicit `chdir` or
   checkout-relative import assumption.
4. Resume from that checkpoint completes one additional optimizer update and
   preserves the resolved lineage.

### B2. Robot-action coordinate mismatch

**Paper/public expectation.** Bridge training and Stage-3 consumption must use
the same physical action representation before the bridge's saved
`action_mean/action_std` transform. The 22D shape alone is not a semantic
contract; see [`DATA.md`](DATA.md) and [`CHECKPOINTS.md`](CHECKPOINTS.md).

**Historical artifact state.** The audited bridge-training cache used a
stride-four difference in the raw action coordinates, while the audited
Stage-3 loader forms stride-four block differences after applying the
dataset-side min-max normalization. The bridge then has its own saved action
standardization. Shape checks alone did not establish equivalence between
those two input coordinates.

**Implemented mitigation.** The public action utility and internal adapter now
first-difference each four-token cumulative block and multiply by
`(action_max - action_min) / 2` before bridge standardization. Golden synthetic
transitions, pinned statistics/layout metadata, and real-asset comparisons
validate the conversion. The bridge remains tied to its recorded LAM and
action metadata. Stage-3 action routing completed in the one-update 2B GPU
smoke; paper-budget action-following evidence is still outstanding.

**Why this blocks training.** Raw-coordinate differences and differences in a
min-max-normalized coordinate system are not interchangeable unless the exact
transform, ordering, units, ranges, and statistics are intentionally aligned.
A numerically valid `(…, 22)` tensor can therefore condition the wrong motion.

**Acceptance evidence.** All of the following must pass:

1. Select and document one canonical Stage-3 input coordinate, including
   dataset/embodiment role, ordered components, units, coordinate frames,
   stride, delta rule, and whether min-max normalization occurs before the
   difference.
2. A golden transition processed independently by the bridge-training adapter
   and Stage-3 loader produces the same bridge input within a declared numeric
   tolerance.
3. The bridge artifact manifest records the coordinate contract, action and
   latent statistic hashes, preprocessing revision, and matching LAM identity.
4. The loader rejects a bridge with absent or mismatched semantic metadata.
5. If the chosen canonical coordinate differs from the existing bridge's
   training coordinate, train and calibrate a new bridge; do not repair the
   mismatch by reshaping or relabeling the old artifact.

### B3. Stage-2 lineage, scope, verdict, and FPS mismatch

**Paper/public expectation.** Stage 2 must initialize the intended ACWM
backbone, consume latents from `stage1_lam_final`, train the declared parameter
scope for 2,000 optimizer updates, and produce the exact state used to
initialize Stage 3. Temporal windows must use verified source timing.

**Historical artifact state.** Earlier Stage-2 configs, checkpoint evidence,
and status labels did not agree on all of the following:

- which Stage-1 LAM and ACWM initialization define the lineage;
- which model parameters are actually trainable;
- whether the artifact is a smoke/progress result or a completed Stage-2 run;
- source FPS, which is hardcoded in the observed loader path rather than
  derived from validated per-source metadata.

**Implemented mitigation.** The real wrapper now generates a LAM registry for
the exact produced Stage-1 checkpoint, decodes windows from manifest records
with validated FPS metadata, records trainable scope and finite update
statistics, and passes the produced Stage-2 checkpoint directly to Stage 3.
The linked one-update H100 run exercised this handoff. It remains a smoke run,
not the required 2,000-update accepted artifact.

**Why this blocks training.** Stage 3 may initialize from a model that was not
debiased with the intended LAM or parameter scope. A hardcoded FPS can also
change transition timing and latent/action alignment while preserving tensor
shapes.

**Acceptance evidence.** All of the following must pass:

1. A Stage-2 provenance manifest identifies `stage1_lam_final`, the ACWM
   initialization, backbone scale, config digest, code revision, and data
   manifest digest.
2. A scope audit records the complete trainable-parameter list before training
   and confirms that only the declared scope changes after one optimizer
   update.
3. Logs and optimizer state prove exactly 2,000 optimizer updates for a
   paper-complete artifact; smoke or partial artifacts remain labeled as such.
4. FPS is read from validated source metadata, or an explicit override is
   recorded and checked against that source. Missing or inconsistent FPS is a
   hard error.
5. A golden window test verifies frame indices, timestamps, transition count,
   and 32D latent alignment.
6. Stage 3 consumes the exact accepted Stage-2 output role, not a similarly
   named or statically selected checkpoint.

### B4. Stage-1 update count and static stage chaining

**Paper/public expectation.** The main Stage-1 artifact is the output after
1,000 optimizer updates with embodiment-centric reconstruction,
action-centric contrast, and latent-space calibration active. Stage 2 must
consume that accepted output.

**Historical artifact state.** The audited 100h Stage-1 research artifact
represents about 150 optimizer updates, not the 1,000-update paper budget.
Earlier downstream launchers also selected parent paths statically.

**Implemented mitigation.** The bundled `pipeline` command discovers the
checkpoint written by the current Stage-1 run, generates the Stage-2 registry
from that exact path, trains and binds a bridge to the same checkpoint, and
then passes the newly written Stage-2 checkpoint and bridge to Stage 3. The
linked H100 smoke verified this dynamic handoff. It does not turn a one-update
checkpoint or the historical partial artifact into the 1,000-update paper
output.

**Why this blocks training.** A short progress checkpoint is not the main paper
checkpoint. Static stage chaining can silently run Stage 2 against an older
LAM even when Stage 1 appears to have completed successfully.

**Acceptance evidence.** All of the following must pass:

1. Optimizer state and logs prove exactly 1,000 Stage-1 optimizer updates for
   the main protocol.
2. Per-objective logging confirms all three CD-LAM objective families are
   finite and active; the mask and primitive-label coverage are recorded.
3. Resume testing shows monotonic optimizer steps and preserves scheduler and
   optimizer state.
4. The pipeline passes the produced `stage1_lam_final` role into Stage 2
   explicitly. Paper configs contain no hidden static checkpoint fallback.
5. The Stage-2 provenance manifest records the digest of the exact Stage-1
   artifact it loaded.

## Release acceptance gates

The training release is accepted only when every gate below has machine-readable
evidence and a failing condition returns a nonzero exit status.

### G1. Configuration and portability

- One schema-validated config resolves every stage role and output location.
- Relative paths have a documented anchor and do not depend on launch CWD.
- `--dry-run` prints the resolved config, stage lineage, trainable scope, data
  manifests, update target, and action-coordinate contract.
- No private paths, credentials, hidden environment defaults, or static
  research-checkpoint fallbacks enter the release config.

### G2. Data and coordinate integrity

- Episode-level train/evaluation overlap is zero.
- One decoded example per stage passes frame, mask, label, FPS, timestamp, and
  transition-alignment checks.
- The bridge-training and Stage-3 golden inputs match in the canonical action
  coordinate.
- Normalization statistics use training data only and carry reproducible
  digests.

### G3. Stage smoke and resume

- Stage 1, Stage 2, and Stage 3 each complete one real optimizer update on the
  declared device path. **Passed for the recorded 2B H100 smoke.**
- Each stage writes a loadable checkpoint and resumes for one more update.
  **Checkpoint writes passed; resume remains open.**
- Trainable-parameter and gradient audits match the declared scope. **The
  one-update smoke recorded finite gradients and parameter counts; a resumed
  update audit remains open.**
- Outputs are finite, and failures cannot be converted into a success verdict
  by the launcher. **Passed for the one-update smoke validators.**

### G4. Paper-budget completion

- Accepted artifacts prove the paper update targets: 1,000 / 2,000 /
  3,000-or-6,000 optimizer updates for Stages 1 / 2 / 3.
- Every accepted artifact is labeled with its role, scale, completion status,
  code revision, config digest, data digest, and parent checkpoint digest.
- Partial, smoke, or exploratory artifacts cannot occupy a paper-complete role.

### G5. Evaluation readiness

- A one-pair rollout consumes the accepted Stage-3 model and matching bridge.
- Zero-action and target-action interventions use their documented reference
  protocols.
- SAM3 mask generation and CoWTracker tracking, or verified compatible cached
  artifacts, produce a finite FDCE result with dependency and validity counts
  recorded.
- Paper-table reproduction is reported only after coverage, failures, and
  protocol identity are explicit.

### G6. Public release hygiene

- Host-side unit tests, config validation, release checks, and package
  installation pass in clean source-only CI.
- GPU stage-smoke tests pass in the gated integration tier.
- Documentation consistently distinguishes core primitives, training-ready
  integration, published assets, and manuscript-reported reference values.
- Checkpoint manifests include both tensor validation and semantic action/LAM
  metadata.

## Current decision

B1 and B2 have host-side validation and a linked one-update 2B GPU path check. The
runtime mitigations for B3 and B4 also passed at smoke scale: the generated
registry, new bridge, and new Stage-2 output were chained in one invocation.
Resume, paper-budget completion, quality evaluation, and accepted artifact
provenance remain open. Therefore:

- the core objective, bridge, and FDCE primitives may remain available for
  testing and integration work;
- protocol configs and manuscript result fixtures remain documentation, not
  proof of a completed training pipeline;
- the bundled real Stage-1/2/3 commands may be used for research and clearly
  labeled smoke or partial artifacts may be distributed;
- promotion to a paper-complete checkpoint role and end-to-end reproduction
  claims remain blocked until G1–G6 pass.
