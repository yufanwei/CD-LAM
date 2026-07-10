# Portable data-contract fixture

`episodes.jsonl` contains two deterministic 49-frame metadata episodes. One is
train and one is test, so split leakage and all four staged data contracts can
be checked without downloading a dataset or model.

The compact `action_sequence` form is intended for tests and examples. Real
metadata may provide an explicit `actions` array with one 22D absolute action
per source frame. Run:

```bash
bash scripts/run.sh data-prepare \
  --input test_data/episodes.jsonl \
  --output outputs/test-data
bash scripts/run.sh data-validate --root outputs/test-data
```

This fixture verifies indexing, FPS, episode splits, action dimensions,
stride-four deltas, and 13-frame Stage 3 alignment. It contains no copyrighted
video or robot dataset payload and does not replace an external adapter's
media-decoding test.
