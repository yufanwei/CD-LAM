# Release manifest

This manifest describes the source release. It intentionally excludes model
weights, datasets, generated rollouts, metric caches, and third-party model
weights.

```text
CD-LAM/
├── README.md
├── Makefile
├── MODEL_CARD.md
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE
├── pyproject.toml
├── configs/
│   ├── paths.example.env
│   ├── pipeline_100h_2b.yaml
│   ├── pipeline_100h_14b.yaml
│   └── eval_paper.yaml
├── docs/
│   ├── ARTIFACTS.md
│   ├── CHECKPOINTS.md
│   ├── DATA.md
│   ├── EVAL_PROTOCOL.md
│   ├── PIPELINE.md
│   ├── TRAINING.md
│   ├── TRAINING_CORRECTNESS.md
│   └── USAGE.md
├── examples/
├── results/
│   ├── README.md
│   └── paper_results.json
├── scripts/
├── src/
├── tests/
└── third_party/
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

The [Hugging Face repository](https://huggingface.co/yufanwei/CD-LAM) is the
declared home for large release assets, but those assets are pending upload.
No checkpoint filename or checksum should be inferred from this manifest.
