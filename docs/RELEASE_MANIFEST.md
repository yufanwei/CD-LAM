# Release manifest

This manifest describes the source release. It intentionally excludes model
weights, datasets, generated rollouts, metric caches, and third-party model
weights.

```text
CD-LAM/
├── .github/        CI and contributing guide
├── configs/        portable pipeline and evaluation templates
├── docs/           method docs, model card, examples, and paper fixtures
├── internal/       release tooling and self-contained data/runtime support
├── scripts/
├── src/
├── tests/          tests plus fixtures/ portable test-set metadata
├── third_party/
├── README.md
├── LICENSE
├── NOTICE
├── CITATION.cff
├── Makefile
├── pyproject.toml
└── requirements.lock
```

## Included

- PyTorch CD-LAM loss, action-transform, and bridge primitives.
- Typed Stage-1/2/bridge/Stage-3 planners and concrete 2B launch wrappers.
- A small source-only ACWM integration overlay pinned by manifest to one
  upstream source revision, plus a complete 737-file staged-tree digest.
- Optimizer-based CPU training smoke with checkpoint, resume, and lineage
  validation.
- AgiBot official-episode materialization, command-preserving conversion,
  split-safe Stage-1/2 manifests, and bridge-cache generation, plus bounded
  official-format EgoDex preparation.
- Portable, relative-path configuration examples for the 100-hour main tier.
- Pinned one-command 2B CUDA environment setup, asset-contract and smoke-test
  gates, and a hash-bound FDCE track scorer.
- Machine-readable Tables I–V copied from the manuscript, with protocol
  qualifiers.
- Documentation of the optional SAM3 and CoWTracker integrations.

## Not included

- CD-LAM or backbone checkpoints.
- Training or evaluation datasets.
- The complete NVIDIA ACWM source tree; the fetch command stages it outside Git
  after explicit license acknowledgement.
- SAM3 or CoWTracker source code and weights.
- Private experiment reports, raw rollout videos, caches, or machine-specific
  launch settings.

The compact Hugging Face upload candidate contains exactly three main 2B
research entries plus the posttraining bridge and action contract. It has been
validated locally, but it has not yet replaced the legacy layout at the
[Hugging Face repository](https://huggingface.co/yufanwei/CD-LAM). After an
immutable compact-release revision is published, its machine-readable
`asset_manifest.json` is authoritative for filenames, tensor roles, sizes,
SHA-256 values, and compatibility metadata. The source-tree manifest does not
duplicate that model-asset index.
