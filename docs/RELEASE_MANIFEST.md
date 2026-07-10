# Release manifest

This manifest describes the source release. It intentionally excludes model
weights, datasets, generated rollouts, metric caches, and third-party model
weights.

```text
CD-LAM/
├── .github/        CI and contributing guide
├── configs/        portable pipeline and evaluation templates
├── docs/           method docs, model card, examples, and paper fixtures
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
- Typed Stage-1/2/bridge/Stage-3 plans and a production adapter interface.
- Optimizer-based CPU training smoke with checkpoint, resume, and lineage
  validation.
- Portable, relative-path configuration examples for the 100-hour main tier.
- Environment, asset-contract, smoke-test, and result-validation entry points.
- Machine-readable Tables I–V copied from the manuscript, with protocol
  qualifiers.
- Documentation of the optional SAM3 and CoWTracker integrations.

## Not included

- CD-LAM, baseline, or backbone checkpoints.
- Training or evaluation datasets.
- SAM3 or CoWTracker source code and weights.
- Private experiment reports, raw rollout videos, caches, or machine-specific
  launch settings.

The [Hugging Face repository](https://huggingface.co/yufanwei/CD-LAM) contains
the released 2B research checkpoint lineages. Its machine-readable
`asset_manifest.json` is authoritative for filenames, tensor roles, sizes,
SHA-256 values, and compatibility metadata. The source-tree manifest does not
duplicate that mutable asset index.
