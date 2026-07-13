# Data preparation contract

This document describes the data boundary required by CD-LAM. The repository
does not redistribute copyrighted datasets, but it does include a
dependency-light JSONL reference builder, an official-schema AgiBot episode
materializer and converter, a bounded official-format EgoDex builder, and the
media/preprocessing support used by the pinned 2B runtime.

## Raw dataset boundary

The JSONL accepted by `data-prepare` is a normalized episode description, not
the raw-dataset entry point. `scripts/download_datasets.py` downloads the
official gated AgiBot Alpha sample or one official Apple EgoDex archive with
explicit license acknowledgement, resume, and archive checks. This repository
also contains an AgiBot Alpha official-episode materializer, raw-to-LeRobot
converter, split-safe Stage-1/2 index/builder, and bridge-cache command; see
[RAW_AGIBOT.md](RAW_AGIBOT.md). It still does
not contain the complete EgoDex transcode/filter stack or the production SAM3
mask-cache generator. Consequently, a successful synthetic
`data-prepare` run does not prove that a fresh raw dataset can be rebuilt into
the paper recipe. The bundled EgoDex indexer closes the official
extraction-to-JSONL boundary for a bounded Stage-1/Stage-2 subset. AgiBot
materialization validates synchronized video, timestamp, state, and independent
publisher command bounds before conversion. The one-command AgiBot path is
`bash run.sh prepare-agibot`; the bounded EgoDex commands are shown
below.

The AgiBot sample downloader pins official dataset commit
`128665c9e0244c45d1cbe5c13f5a4706afd24f27` and verifies both the
7,097,989,120-byte size and SHA-256 of `sample_dataset.tar`. It writes
`agibot_download.json`; the materializer consumes that record automatically
when it sits beside the extracted `sample_dataset/` directory. Local data
without this record remains usable but is not presented as publisher-verified
revision provenance.

Official sources:

- AgiBotWorld Alpha: <https://github.com/OpenDriveLab/Agibot-World> and
  <https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha>.
- EgoDex: <https://github.com/apple/ml-egodex>; the repository links the five
  training ZIPs, native test ZIP, and extra-data ZIP on Apple's CDN. Direct
  examples are the official [Part 2 archive](https://ml-site.cdn-apple.com/datasets/egodex/part2.zip)
  and [native test archive](https://ml-site.cdn-apple.com/datasets/egodex/test.zip).

Paths under `/data` below are example mount points. If `/data` is not writable,
use the `$CDLAM_DATA_ROOT` setup from the top-level README or replace `/data`
with an absolute directory owned by your user.

```bash
.venv/bin/python scripts/download_datasets.py links
.venv/bin/python scripts/download_datasets.py agibot-sample \
  --output /data/raw/agibot-alpha --accept-license --extract
.venv/bin/python scripts/download_datasets.py egodex \
  --part part2 --output /data/raw/egodex --accept-license --extract
```

The repository provides both a fail-closed provenance preflight and a
bounded raw-clip adapter test. The adapter embeds explicitly listed MP4 clips
and builds Stage-1/Stage-2 train/eval manifests; it does not replace the full
paper data pipeline. See [RAW_EGODEX.md](RAW_EGODEX.md) for automatic EgoDex
index generation and [RAW_SUBSET.md](RAW_SUBSET.md) for the normalized JSONL
contract, output tree, and cleanroom command.

For EgoDex Part 2, the direct bounded path is:

```bash
.venv/bin/python scripts/index_egodex.py \
  --root /data/raw/egodex/extracted/part2 \
  --part part2 \
  --output /data/prepared/egodex-part2-bounded.jsonl \
  --eval-fraction 0.10 --seed 42 --max-clips 32
PYTHONPATH=internal/vendor/scale_support \
  .venv/bin/python internal/runtime/audit_raw_splits.py \
  --input /data/prepared/egodex-part2-bounded.jsonl
.venv/bin/python internal/runtime/build_raw_subset.py \
  --input /data/prepared/egodex-part2-bounded.jsonl \
  --output /data/prepared/cdlam-stage12-subset
```

The bounded indexer chooses complete physical-session groups and preserves
both train and evaluation labels; it does not take the first 32 files or split
one session. Omit `--max-clips` only when producing a provenance index that
will not be passed to the bounded subset builder. That builder accepts at most
256 explicitly listed clips and defaults to 32.

Use the provenance-only preflight for rows emitted by another converter:

```bash
PYTHONPATH=internal/vendor/scale_support \
  .venv/bin/python internal/runtime/audit_raw_splits.py \
  --input /path/to/raw_provenance.jsonl
```

The accepted provenance rows are intentionally small:

```json
{"dataset":"agibot_alpha","source_id":"327-648642-000","split":"train"}
{"dataset":"egodex","episode_id":"egodex_test_arrange_topple_dominoes_0","part":"test","session_name":"2025-03-04_14-13-51.mov","split":"test"}
```

For AgiBot Alpha, a raw segment directory is
`<task>-<episode>-<segment>`, but the physical isolation key is
`agibot_alpha:<task>:<episode>`. Every segment from that episode must receive
the same split. Bridge caches store the physical key in `episode_id` and retain
the raw segment in `segment_id`.

The one-command AgiBot output also contains
`stage12/stage1/lam_pair_{train,eval}.parquet` and
`stage12/stage2/wm_{train,eval}_manifest.parquet`. Its bounded selector retains
complete physical-episode groups and guarantees nonempty train/evaluation
groups before embedding media into the portable shard. Exact paths are written
to `prepare_summary.json` for direct transfer into `configs/runtime.json`.

For EgoDex, the episode identity is `<part>:<task>:<index>`, while the physical
isolation key is `<part>:<session_name>` from the HDF5 root attributes. Several
clips can share one `session_name`. Apple's native `test` part is immutable
holdout data and must remain `split: "test"`; it cannot be reassigned to either
`train` or `eval`.

A true rebuild therefore needs, at minimum:

- AgiBot Alpha `train/<task>-<episode>-<segment>/head_color.mp4`, the matching
  `proprio_stats.h5`, and `task_info/task_<task>.json`. The HDF5 must contain
  `action/{joint,effector,head,waist}/position` and
  `action/robot/velocity`; state arrays are observations and are never used as
  command substitutes;
- EgoDex `<part>/<task>/<index>.mp4` plus the matching HDF5 file, with the
  native part and `session_name` preserved through packing;
- a converter-owned provenance sidecar that survives any integer ID hashing;
- the pinned SAM3 source, accepted gated weights, and a mask generator that
  writes `masks_sam3.npz`, `frame_to_mask_idx.npy`, and mask metadata using the
  same crop/resize path as Stage 1.

The 100h and 1000h paper recipes contain additional datasets. Rebuilding only
AgiBot and EgoDex is a valid adapter test or subset experiment, not a
reproduction of either complete paper recipe.

```bash
bash run.sh data-prepare \
  --input tests/fixtures/episodes.jsonl \
  --output outputs/test-data
bash run.sh data-validate --root outputs/test-data
```

Production episode JSONL may provide one explicit 22D absolute `actions` row
per source frame. The compact `action_sequence` form in `tests/fixtures` is
only a deterministic fixture generator. The command produces Stage-1 pairs, Stage-2
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
LAM extracts 32D latents from transitions inside that window, and the bundled
2B runtime feeds those latents directly into the ACWM conditioning path. The
Stage-2 LAM, any cached 32D latents, the later bridge, and Stage-3
initialization must all refer to the same latent space. A custom backbone
adapter must preserve this contract.

## Bridge training and Stage 3: paired robot actions

The paper's AgiBot Alpha action vector has 22 ordered components:

```text
action/joint/position 14
+ action/effector/position 2
+ action/head/position 2
+ action/waist/position 2
+ action/robot/velocity 2
= 22 dimensions
```

The converter copies these publisher command arrays without implicit unit
conversion. It reads the state arrays separately for the observation state and
fails when a command array is missing or misaligned; it does not derive actions
from state, rescale the effector command, or fill base velocity with zeros.

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
`normalized_block_anchor_to_raw_adjacent` in `cd_lam.data.action`; the bundled
Stage-3 wrapper applies this conversion before reduced-precision casting.

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
