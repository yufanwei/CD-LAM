# Contributing

Thank you for helping improve CD-LAM.

## Development setup

```bash
bash setup.sh --accept-base-license
bash run.sh test
bash run.sh smoke
bash run.sh release-check
```

The GitHub-hosted source tests remain deterministic and do not require model
weights, network access, SAM3, CoWTracker, private datasets, or GPU acceptance.
The supported newcomer bootstrap separately requires the documented live GPU
contract and performs a CUDA optimizer smoke.

## Pull requests

- Keep public paths relative and portable. Do not commit credentials, local
  environment files, private paths, caches, checkpoints, datasets, or generated
  rollout videos.
- Add tests for behavior changes and run the release checker.
- Preserve the 22D-to-32D bridge contract and reject incomplete normalization
  bundles rather than guessing defaults.
- Keep Stage-1, Stage-2, Stage-3, and intervention protocols distinct in code
  and reporting.
- Update `docs/results/paper_results.json` only when correcting against an
  authoritative manuscript revision; include the source table and protocol
  impact in the pull-request description.
- Do not add checkpoint filenames, hashes, benchmark claims, or public asset
  availability until the artifact exists and has been verified.

## Optional integrations

Changes involving SAM3, CoWTracker, datasets, or an ACWM backbone must preserve
their license boundaries. Do not copy third-party source or weights into this
repository unless redistribution is explicitly permitted and approved.

## Reporting issues

Include the CD-LAM revision, Python/PyTorch versions, platform, command, minimal
config, and full traceback. For FDCE issues also include evaluation resolution,
SAM3 and CoWTracker revisions, valid-track counts, and whether the failure is
in segmentation, tracking, or aggregation. Never attach private data or model
weights to a public issue.
