# AgiBot Alpha raw-data conversion

`scripts/materialize_agibot_alpha.py` converts official physical episodes into
frame-aligned action segments. `internal/runtime/convert_agibot_alpha.py` then
converts those segments into the LeRobot-style inputs used by CD-LAM Stage 3
and action-bridge training. A split-safe indexer and the bundled raw-subset
builder produce the matching Stage-1/Stage-2 train/evaluation manifests.
Neither tool has private path defaults.

This converter is deliberately scoped to AgiBot Alpha robot actions. It does
not generate Stage-1 SAM3 masks, convert EgoDex, or reproduce the additional
datasets in the 100h and 1000h paper mixtures.

## Official download

Use only the publisher's [AgiBot World repository](https://github.com/OpenDriveLab/Agibot-World)
and [official gated Alpha dataset](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha).
The publisher describes about 8.5 TB of Alpha dataset content; the Hugging Face
repository footprint may be larger. The official dataset also provides a
roughly 7.1 GB `sample_dataset.tar`. This release pins dataset commit
`128665c9e0244c45d1cbe5c13f5a4706afd24f27`, archive size 7,097,989,120
bytes, and archive SHA-256
`131c6f99ebe6900e93d56be9f0cbe46f2cff286b8d9102b8d3e01d25f7cebe5e`:

```bash
export CDLAM_DATA_ROOT="${CDLAM_DATA_ROOT:-$PWD/data}"
mkdir -p "$CDLAM_DATA_ROOT/raw" "$CDLAM_DATA_ROOT/prepared"
.venv/bin/hf auth login
.venv/bin/python scripts/download_datasets.py agibot-sample \
  --output "$CDLAM_DATA_ROOT/raw/agibot-alpha" \
  --accept-license --extract
```

The dataset is gated. Accept the terms in a browser before running the command
with the same Hugging Face account. A `403` means that account has not been
approved. The downloader does not request access or bypass the gate.
Successful extraction writes
`$CDLAM_DATA_ROOT/raw/agibot-alpha/sample_dataset/`; the verified source record
is `agibot_download.json` beside it. That directory is an
official episode-layout sample, not the segmented converter input documented
below. Public metadata does not prove that every sample episode contains all
five command arrays required by the 22D action contract. Treat it as a format
inspection download; the materializer fails closed when commands are absent.

The official full repository stores episode-level archives under
`observations/<task>/`, shared-range archives under `proprio_stats/` and
`parameters/`, and labels under `task_info/`. Its own task-selective procedure
is equivalent to:

```bash
git lfs install
git init AgiBotWorld-Alpha
cd AgiBotWorld-Alpha
git remote add origin \
  https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
git sparse-checkout init
git sparse-checkout set \
  observations/327 task_info/task_327.json scripts proprio_stats parameters
git pull origin main
```

This can still select more than 100 GB because proprioception and parameters
are stored in range archives. Inspect the repository file sizes before pulling.
The official archive layout is episode-level. Extract selected archives before
running the materializer; do not point it at `.tar` files. It applies each
publisher `[start_frame, end_frame)` interval to `head_color.mp4`, timestamp,
and every frame-aligned HDF5 array. It filters and rebases `action/*/index`
arrays and rejects every other ambiguous length.

The publisher also provides `scripts/convert_to_lerobot.py` in its dataset
repository. The reviewed schema reference is Hugging Face revision
`128665c9e0244c45d1cbe5c13f5a4706afd24f27`. The bundled downloader records and
revalidates that revision and archive digest automatically. Manually acquired
local inputs remain supported, but a caller-supplied `--source-revision`
without the verified source record is explicitly labeled caller-attested.

## One-command preparation

```bash
bash run.sh prepare-agibot \
  --raw-root "$CDLAM_DATA_ROOT/raw/agibot-alpha/sample_dataset" \
  --output-root "$CDLAM_DATA_ROOT/prepared/agibot-alpha" \
  --max-episodes 32
```

This runs five fail-closed links without a model runtime: episode
materialization, corrected LeRobot conversion, physical-episode Stage-1/2
indexing, portable Stage-1/2 subset construction, and bridge-cache generation.
Use `--dry-run` to inspect every command. If deterministic train/eval hashing
is empty, it fails and asks for more complete physical episodes; segments are
never split apart.

The final tree is directly routable into `configs/runtime.json`:

```text
agibot-alpha/
├── segmented/
├── lerobot/
├── stage12_index.jsonl
├── stage12/
│   ├── stage1/lam_pair_{train,eval}.parquet
│   └── stage2/wm_{train,eval}_manifest.parquet
├── bridge/alpha_bridge_cache.npz
└── prepare_summary.json
```

`prepare_summary.json` records the exact output paths. The Stage-1/2 builder
copies bounded video bytes into a portable shard and caps selection at complete
physical-episode groups; it never splits one recording to meet a clip limit.

The official input is
`observations/<task>/<episode>/videos/head_color.mp4`,
`proprio_stats/<task>/<episode>/proprio_stats.h5`, and
`task_info/task_<task>.json`. Required commands are
`action/{joint,effector,head,waist}/position` and `action/robot/velocity`.

## Segmented intermediate

The input root must have this shape:

```text
agibotworld_alpha/
├── task_info/
│   └── task_<task>.json
└── train/
    └── <task>-<physical_episode>-<segment>/
        ├── head_color.mp4
        └── proprio_stats.h5
```

The single `.venv` includes `numpy`, `h5py`, `pandas`, `pyarrow`, and PyAV.
PyAV checks the video stream, frame count, and 30 FPS contract without decoding
the full video.

## Convert once

Run the converter from the CD-LAM bundle root. The output must be a new path;
`--overwrite` is required to replace an earlier conversion.

```bash
DATA_PYTHON=.venv/bin/python

"$DATA_PYTHON" internal/runtime/convert_agibot_alpha.py \
  --raw-root "$CDLAM_DATA_ROOT/raw/agibot-alpha/segmented" \
  --output "$CDLAM_DATA_ROOT/prepared/cdlam-agibot-alpha"
```

Videos are copied by default. The resulting dataset can therefore be moved or
synced without the raw tree. For a local, non-portable experiment, opt in to
absolute source links:

```bash
"$DATA_PYTHON" internal/runtime/convert_agibot_alpha.py \
  --raw-root "$CDLAM_DATA_ROOT/raw/agibot-alpha/segmented" \
  --output "$CDLAM_DATA_ROOT/prepared/cdlam-agibot-alpha-linked" \
  --video-mode symlink
```

For a bounded acceptance run, supply a text file containing one raw source ID
per line, for example `392-679233-012`:

```bash
"$DATA_PYTHON" internal/runtime/convert_agibot_alpha.py \
  --raw-root "$CDLAM_DATA_ROOT/raw/agibot-alpha/segmented" \
  --output "$CDLAM_DATA_ROOT/prepared/cdlam-agibot-alpha-smoke" \
  --clip-list "$CDLAM_DATA_ROOT/prepared/agibot-smoke.txt"
```

The default error policy is fail-closed. `--on-error skip` is available for an
explicit salvage run; every skipped source and error is then recorded in
`build_summary.json`. `--skip-video-check` is only for metadata plumbing tests,
not a production conversion.

## Output contract

One invocation writes disjoint roots and portable bridge-builder YAML files:

```text
cdlam-agibot-alpha/
├── train/
│   ├── data/
│   ├── videos/
│   └── meta/
├── eval/
├── test/
├── _dataset_paths_train.yaml
├── _dataset_paths_eval.yaml
├── _dataset_paths_test.yaml
├── _dataset_paths_all.yaml
├── provenance.jsonl
└── build_summary.json
```

The physical split key is `agibot_alpha:<task>:<episode>`. All action segments
from one physical recording receive the same split. Three deterministically
selected task IDs form the unseen-task `test` split; 10 percent of remaining
physical episodes form the seen-task `eval` split; the rest form `train`.
`provenance.jsonl` retains the task, physical episode, segment, split rule, raw
relative directory, output episode index, and frame count.

Some official segments contain one terminal proprio record after the final
decodable camera frame. The converter drops exactly that final record and
records `alignment_policy=dropped_terminal_proprio`. Larger video/proprio
mismatches are ambiguous and fail closed.

Do not concatenate these roots and ask the external loader to make a trailing
5-percent frame split. Formal evaluation must use the already isolated `eval`
or `test` root. `_dataset_paths_all.yaml` exists for conversion audits and
bounded end-to-end acceptance tests; it must not be used as a training recipe.

The absolute action is copied from publisher command arrays without implicit
unit conversion:

```text
action/joint/position 14
+ action/effector/position 2
+ action/head/position 2
+ action/waist/position 2
+ action/robot/velocity 2
= 22 dimensions
```

## Audit and build the bridge cache

Audit physical split isolation before sampling transitions:

```bash
PYTHONPATH=internal/vendor/scale_support \
  "$DATA_PYTHON" internal/runtime/audit_raw_splits.py \
  --input "$CDLAM_DATA_ROOT/prepared/cdlam-agibot-alpha/provenance.jsonl"
```

Then pass the generated training YAML directly to the bridge-cache builder.
Relative dataset paths are resolved beside the YAML file, so moving the whole
conversion tree does not invalidate it.

```bash
PYTHONPATH=internal/vendor/scale_support \
  "$DATA_PYTHON" \
  internal/vendor/scale_support/Scale/common/build_alpha_bridge_cache.py \
  --dataset-yaml "$CDLAM_DATA_ROOT/prepared/cdlam-agibot-alpha/_dataset_paths_train.yaml" \
  --out "$CDLAM_DATA_ROOT/prepared/cdlam-agibot-alpha/bridge/alpha_bridge_cache.npz" \
  --n-episodes 4000 \
  --pairs-per-episode 8 \
  --stride 4
```

The bridge builder independently creates an episode-disjoint calibration
split inside the training pool. Its `episode_id` array stores the physical key,
and `segment_id` stores the raw `<task>-<episode>-<segment>` provenance.

For Stage 1 and Stage 2, use the four paths reported under `stage12/`. For
Stage 3, set the runtime dataset path to the generated `lerobot/train/` root or
to the single entry in `lerobot/_dataset_paths_train.yaml`. Use the reported
bridge NPZ for bridge training. The matching pinned `AgiBot_stats.json`,
modality contract, Stage-1 LAM, and bridge checkpoint still have to move
together as described in [DATA.md](DATA.md) and [CHECKPOINTS.md](CHECKPOINTS.md).
