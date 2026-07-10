# Data preparation contract

This document describes the data boundary required by CD-LAM. The public core
does not ship copyrighted datasets or a production media dataloader. It does
include a dependency-light JSONL reference builder and validator, while the
external LAM/ACWM adapter remains responsible for media decoding.

```bash
bash scripts/run.sh data-prepare \
  --input test_data/episodes.jsonl \
  --output outputs/test-data
bash scripts/run.sh data-validate --root outputs/test-data
```

Production episode JSONL may provide one explicit 22D absolute `actions` row
per source frame. The compact `action_sequence` form in `test_data` is only a
deterministic fixture generator. The command produces Stage-1 pairs, Stage-2
windows, bridge pairs, and Stage-3 windows with aligned transition indices.

## Rules shared by every stage

1. Split by episode before sampling transitions or windows. Neighboring frames
   from one episode must never cross train/evaluation boundaries.
2. Store stable sample and episode IDs, source frame indices or timestamps,
   FPS, split, and the exact crop/resize policy.
3. Keep frame/action alignment and temporal stride explicit. A tensor with the
   right final dimension can still have the wrong physical meaning.
4. Compute normalization statistics from the training split only.
5. Record the matching LAM checkpoint identity for every latent cache and
   bridge. Changing the LAM invalidates cached latents and bridge calibration.

## Stage 1: transition pairs for LAM fine-tuning

Stage 1 needs ordered video pairs. A reference pair index preserves at least:

| field | meaning |
|---|---|
| `sample_id`, `episode_id`, `split` | stable identity and episode-level split |
| source reference and row index | enough information for the adapter to decode the clip |
| `frame_i`, `frame_j` | ordered transition indices; equal indices mark an identity transition |
| `primitive` | one of the 12 coarse verb categories, or `-1`/empty when unlabeled |
| foreground-mask reference | soft embodiment/interacted-object mask aligned to each decoded frame |

The paper reference path center-crops video to 4:3, uses 480x640 world-model
frames, and produces 240x320 LAM inputs. Exact embodiment-centric weighting
requires SAM3 masks or protocol-compatible precomputed masks. Falling back to
a full-frame mask is useful for plumbing tests but is not paper-equivalent.

## Stage 2: video windows with direct latent conditioning

Stage 2 uses action-unlabeled video and **does not use the robot bridge**. A
reference clip manifest preserves:

```text
sample_id, episode_id, split, source_reference,
start_frame, stop_frame, clip_nframes, fps
```

Clips must contain at least 13 usable frames for the paper window. The selected
LAM extracts 32D latents from transitions inside that window, and the external
ACWM adapter consumes those latents directly. The Stage-2 LAM, any cached 32D
latents, the later bridge, and Stage-3 initialization must all refer to the same
latent space.

## Bridge training and Stage 3: paired robot actions

The paper's AgiBot Alpha action vector has 22 ordered components:

```text
arm 14 + grippers 2 + head 2 + waist 2 + base 2
```

The bridge-training target is aligned to a fixed source stride of four frames:

```text
action_22 = absolute_action[t + 4] - absolute_action[t]
```

The reference bridge cache contains:

| array | shape and dtype | meaning |
|---|---|---|
| `frames` | `(N, 2, 240, 320, 3)`, `uint8` | aligned LAM frame pair |
| `action_22` | `(N, 22)`, `float32` | stride-4 action delta |
| `episode_id` | `(N,)` | episode-level isolation key |
| `split` | `(N,)` | deterministic episode-level train/eval split |

An identity pair `(t, t)` with a zero action delta may be included as the
zero-transition anchor. The bridge learns action-side statistics from the
training split and targets latents produced by the matching LAM; its artifact
must retain `action_mean`, `action_std`, `zm`, and `zsd` with the MLP weights.

The reference Stage-3 loader starts from the dataset's min-max-normalized
action sequence and emits cumulative differences inside four-token blocks. A
13-frame world-model window yields 12 transition conditions. Before invoking
the raw adjacent-delta bridge, first-difference within each block and restore
raw units with `(action_max - action_min) / 2`. The tested implementation is
`normalized_block_anchor_to_raw_adjacent` in `cd_lam.data.action`.

Do not feed an absolute command, a differently ordered 22D vector, the loader's
unconverted cumulative delta, or a packed multi-step vector into this bridge.
Train and calibrate a new bridge for another robot, cadence, unit convention,
or action representation.

## Minimal validation before a distributed run

- Decode one Stage-1 pair and verify frame order, crop, mask alignment, and
  identity-pair behavior.
- Decode one 13-frame Stage-2 window and verify 12 finite 32D latents from the
  intended LAM checkpoint.
- For Stage 3, print the 22 component names, source indices, delta, and bridge
  output for one transition; verify output shape `(32,)` and finite values.
- Confirm the bridge checkpoint and Stage-2/Stage-3 models share the same LAM
  identity and normalization metadata.
- Run train/evaluation episode-overlap checks before writing any cache.

SAM3 and CoWTracker remain external. SAM3 creates the masks needed for exact
Stage-1 weighting and FDCE; CoWTracker creates the point tracks used by FDCE.
Neither is required when compatible masks/tracks have already been computed.
