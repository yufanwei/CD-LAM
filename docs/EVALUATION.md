# Evaluation

CD-LAM separates rollout generation from metric reduction. A rollout backend
must first produce videos under fixed model, action, context, seed, and data
population identities. SAM3 and CoWTracker then produce protocol-compatible
foreground tracks. The public scorer consumes only those tracks, so metric
reduction is deterministic and does not silently download a model.

SAM3 and CoWTracker are optional external source and weight caches. They are
needed only when generating new masks or tracks, are governed by their upstream
licenses, and do not create another CD-LAM environment. Compatible precomputed
tracks can be scored directly from the supported `.venv`.

## One-command FDCE scoring

Each input is an NPZ archive with exactly these arrays:

| key | shape | required |
|---|---|---:|
| `generated_tracks` | `(49, N_generated, 2)` | yes |
| `reference_tracks` | `(49, N_reference, 2)` | yes |
| `generated_visibility` | `(49, N_generated)` | no |
| `reference_visibility` | `(49, N_reference)` | no |

Coordinates are pixels at the recorded evaluation resolution. The default
paper-aligned gate accepts at most 16 tracks per side, requires frame zero plus
48 rollout frames, removes tracks below 80% visibility, and applies the
symmetric Chamfer reduction after computing each fixed track-pair displacement
cost over time.

```bash
bash run.sh score-fdce \
  --tracks evaluation/tracks/sample-000.npz evaluation/tracks/sample-001.npz \
  --output evaluation/fdce.json
```

The output records every input SHA-256, per-sample directional terms and
valid-track counts, followed by an unweighted sample mean and median. It never
records a machine-local absolute path. Duplicate files, unexpected NPZ keys,
wrong frame counts, too many anchors, unsafe or oversized archives, invalid
visibility, or incomplete track comparisons fail before a result is written.

Use `--expected-frames 0` only for an explicitly labeled non-paper diagnostic.
Changing visibility thresholds, track limits, resolution, population, or
aggregation creates a different protocol and must not be compared as a paper
reproduction.

## What is tested

`bash run.sh check` tests analytical FDCE cases, the fixed-pair-before-
Chamfer reduction order, visibility filtering, archive validation, aggregation,
input hashing, and atomic report writing. `validate-results` separately checks
that the manuscript tables were transcribed consistently.

The release does not present the legacy private rollout scripts as portable.
Exact trained-model video reproduction still needs the compatible external
NVIDIA base model and tokenizer/text encoder assets, the pinned `.venv` GPU runtime,
the selected held-out population, and a rollout adapter for that base model.
Do not substitute source-check success or track scoring for a rollout-quality
claim. Stage 1, Stage 2, and Stage 3 already run their own held-out loss and
conditioning diagnostics during training; those are execution gates, not
FDCE/PSNR paper-table reproduction.
