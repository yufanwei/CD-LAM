# Bounded raw-data subset builder

`internal/runtime/build_raw_subset.py` is a cleanroom adapter test for explicit
raw AgiBot Alpha and Apple EgoDex MP4 clips. It creates one embedded-video
Parquet shard plus separate Stage-1 pair and Stage-2 window manifests for
`train` and `eval`.

This is not the paper recipe builder. It does not discover an entire dataset,
download data, filter EgoDex, generate SAM3 masks, build action/bridge caches,
or reproduce the 100h/1000h source mixture. Its defaults reject more than 32
input clips, and the hard maximum is 256 clips.

Do not pass an unbounded EgoDex training-part index to this command. Build the
input with `scripts/index_egodex.py --max-clips 32`; that indexer selects whole
physical-session groups, preserves the existing train/evaluation labels, and
fails instead of slicing a session to fit the bound. See
[RAW_EGODEX.md](RAW_EGODEX.md) for the complete archive-to-index command.

## Runtime requirements

The command needs Python 3.10 or newer plus `pyarrow` and `av`. `h5py` is only
needed when `session_name` is read from an EgoDex HDF5 metadata file. These
packages are data-adapter dependencies; they are not part of the lightweight
public-core environment.

Install and verify the data tools inside the repository environment before
reading a large clip:

```bash
.venv/bin/python -m pip install -e '.[data]'
.venv/bin/python -c 'import av, h5py, pyarrow; print(av.__version__, h5py.__version__, pyarrow.__version__)'
```

## Explicit input contract

Paths are resolved relative to the JSONL file unless they are absolute. Every
row must declare its split. Do not split after pair/window sampling.

```json
{"dataset":"agibot_alpha","source_id":"327-648642-000","video_path":"raw/agibot/327-648642-000/head_color.mp4","task_name":"place fruit","split":"train"}
{"dataset":"egodex","episode_id":"egodex_part2_basic_pick_place_1094","video_path":"raw/egodex/part2/basic_pick_place/1094.mp4","session_name":"2025-02-01_12-00-00.mov","split":"eval"}
{"dataset":"egodex","episode_id":"egodex_test_arrange_topple_dominoes_0","video_path":"raw/egodex/test/arrange_topple_dominoes/0.h264.mp4","metadata_h5":"raw/egodex/test/arrange_topple_dominoes/0.hdf5","split":"test"}
```

For AgiBot Alpha, `source_id` must be
`<task>-<physical_episode>-<segment>`. All segments sharing the first two
components must have the same split.

For EgoDex, `episode_id` must be `egodex_<part>_<task>_<index>` and
`session_name` is the physical recording key. It may be given directly or read
from the root `session_name` attribute of `metadata_h5`. Multiple clips from
one session must have the same split. Native Apple `test` clips must retain
`split: "test"`; their provenance is recorded, but their bytes never enter the
train/eval shard or manifests.

Optional fields include `clip_id`, integer `task_id`, `primitive`,
`primitive_raw`, `step_starts`, `step_ends`, `step_actions`, and `step_skills`.
Frame-index annotations are range checked after the MP4 is decoded.

## Build and inspect

```bash
.venv/bin/python internal/runtime/build_raw_subset.py \
  --input /path/to/raw_clips.jsonl \
  --output /path/to/prepared-raw-subset
```

The command fails if the output already exists, a physical recording crosses
splits, a native EgoDex test row is relabeled, either train or eval is absent,
a clip has fewer than 13 frames, or configured byte/clip bounds are exceeded.
It writes atomically after all source clips pass validation.

```text
prepared-raw-subset/
  build_report.json
  provenance.jsonl
  shards/raw-subset-00000.parquet
  stage1/lam_pair_train.parquet
  stage1/lam_pair_eval.parquet
  stage2/wm_train_manifest.parquet
  stage2/wm_eval_manifest.parquet
```

The shard contains the original MP4 bytes, sequential frame timestamps in
microseconds, measured FPS, source/session identifiers, source hashes, and
split/group keys. Stage-2 `video_path` values already use
`<shard_path>::<row_index>`, which is the bundled shard decoder contract.

Stage-1 rows intentionally set `mask_policy=full_frame_plumbing_only`, leave
mask paths empty, and set `paper_equivalent_mask=false`. This supports decoder,
loader, and model-plumbing checks only. It must not be used to report the
paper's mask-weighted Stage-1 result, FDCE, or reproduction status. Generate
protocol-compatible SAM3 masks before any quality claim.

`build_report.json` records input/output hashes, split groups, sampling bounds,
excluded native-test counts, and the non-paper-equivalent mask warning. Treat
that report as the audit entry point for a representative cleanroom run.
