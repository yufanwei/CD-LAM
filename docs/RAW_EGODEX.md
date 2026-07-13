# EgoDex raw indexing

This is the bounded path from an extracted official Apple EgoDex archive to
the normalized clip index consumed by CD-LAM's Stage-1/Stage-2 subset builder.
It does not redistribute EgoDex, transcode media, generate SAM3 masks, or
construct the complete paper data mixture.

## Official source and terms

Use the publisher's repository and download links:
<https://github.com/apple/ml-egodex>. The dataset is distributed under
CC BY-NC-ND. Review the current publisher terms before downloading or
processing it; the CD-LAM Apache-2.0 license does not replace them.

Apple provides five training parts of roughly 300 GB each, a roughly 16 GB
native test archive, and roughly 200 GB of extra data. Use the native test
archive only to inspect the format. A real CD-LAM train/evaluation adapter run
needs a non-test part with at least two physical `session_name` groups.

```bash
export CDLAM_DATA_ROOT="${CDLAM_DATA_ROOT:-$PWD/data}"
mkdir -p "$CDLAM_DATA_ROOT/raw" "$CDLAM_DATA_ROOT/prepared"

# Format inspection only.
.venv/bin/python scripts/download_datasets.py egodex \
  --part test --output "$CDLAM_DATA_ROOT/raw/egodex" \
  --accept-license --extract

# Source clips for a bounded train/evaluation subset.
.venv/bin/python scripts/download_datasets.py egodex \
  --part part2 --output "$CDLAM_DATA_ROOT/raw/egodex" \
  --accept-license --extract
```

Archives are retained as `$CDLAM_DATA_ROOT/raw/egodex/<part>.zip`. Each part is
extracted under `$CDLAM_DATA_ROOT/raw/egodex/extracted/<part>/`, so downloading
another part does not overwrite a previous extraction. Plan storage for both
the archive and its extracted files: more than 16 GB for `test` and more than
300 GB for `part2`; the exact extracted size may be larger. Delete an archive
only after validating the extraction and recording its provenance.

## Accepted tree

The indexer scans recursively, so one publisher-created wrapper directory is
allowed. Each clip must have exactly one HDF5 metadata file and one video in
the same task directory:

```text
extracted/part2/
└── optional-wrapper/
    └── basic_pick_place/
        ├── 0.hdf5
        ├── 0.mp4
        ├── 1.hdf5
        └── 1.h264.mp4
```

The numeric basename is the clip index. Supported videos are `<index>.mp4`
and `<index>.h264.mp4`. Every HDF5 file must contain a nonempty scalar root
attribute named `session_name`. Missing counterparts, both video encodings for
one clip, duplicate task/index identities, nonnumeric indices, and unreadable
session metadata fail closed.

## Build a deterministic index

```bash
.venv/bin/python scripts/index_egodex.py \
  --root "$CDLAM_DATA_ROOT/raw/egodex/extracted/part2" \
  --part part2 \
  --output "$CDLAM_DATA_ROOT/prepared/egodex-part2-bounded.jsonl" \
  --eval-fraction 0.10 \
  --seed 42 \
  --max-clips 32
```

For non-test parts, the split unit is `<part, session_name>`, never an
individual clip. Sessions are ranked by SHA-256 over the documented policy,
seed, part, and session name. The closest requested evaluation count is
clamped so both train and evaluation are nonempty. Fewer than two sessions
therefore fails before writing an index. Reordering files does not change the
assignment.

`--max-clips 32` then chooses whole session groups deterministically while
retaining nonempty train and evaluation splits. It never truncates a session
or takes an arbitrary first-files slice. This bound matches the subset
builder's default. Omit the option only to create a full provenance index for
separate inspection; a full Part 2 index is normally too large for the bounded
builder, whose absolute limit is 256 clips.

For `--part test`, every row remains `split=test`. Native test rows can be
recorded in provenance, but the bounded subset builder intentionally excludes
their media from train/evaluation outputs. A test-only index cannot satisfy a
training subset and should not be passed to the builder by itself.

The output is atomic and is not overwritten unless `--overwrite` is supplied.
Media and HDF5 paths are relative to the JSONL location, which lets the index
and source tree move together.

## Optional primitive labels

The indexer never guesses a coarse action label from a task name. To use
caption-derived primitive supervision, provide an explicitly reviewed JSON
object:

```json
{
  "assemble_disassemble_furniture_bench_chair": "insert_remove",
  "basic_pick_place": "pick_place"
}
```

```bash
.venv/bin/python scripts/index_egodex.py \
  --root "$CDLAM_DATA_ROOT/raw/egodex/extracted/part2" \
  --part part2 \
  --output "$CDLAM_DATA_ROOT/prepared/egodex-part2-bounded.jsonl" \
  --eval-fraction 0.10 --seed 42 \
  --max-clips 32 \
  --primitive-map "$CDLAM_DATA_ROOT/config/egodex-primitives.json"
```

Unknown tasks remain unlabeled. The mapping is experiment input and should be
versioned with the prepared-data report rather than treated as publisher
ground truth.

## Build and validate the bounded subset

```bash
PYTHONPATH=internal/vendor/scale_support \
  .venv/bin/python internal/runtime/audit_raw_splits.py \
  --input "$CDLAM_DATA_ROOT/prepared/egodex-part2-bounded.jsonl"

.venv/bin/python internal/runtime/build_raw_subset.py \
  --input "$CDLAM_DATA_ROOT/prepared/egodex-part2-bounded.jsonl" \
  --output "$CDLAM_DATA_ROOT/prepared/cdlam-stage12-subset"
```

The builder decodes every selected MP4, requires at least 13 frames, embeds
the original video bytes into a portable Parquet shard, and writes separate
Stage-1 and Stage-2 train/evaluation manifests. It records a full-frame mask
policy for plumbing only. Generate protocol-compatible SAM3 masks before
claiming paper-equivalent Stage-1 supervision or FDCE reproduction.

See [RAW_SUBSET.md](RAW_SUBSET.md) for output schemas and size bounds, and
[DATA.md](DATA.md) for the contracts shared with AgiBot, the bridge, and
Stage 3.
