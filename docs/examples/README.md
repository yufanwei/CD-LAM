# Examples

Run a one-command source-release bootstrap:

```bash
bash scripts/bootstrap.sh
```

Or run the checks independently:

```bash
bash scripts/run.sh doctor --strict
bash scripts/run.sh smoke
bash scripts/run.sh validate-results
```

Run both supported conditioning routes without a model checkpoint:

```bash
.venv/bin/python docs/examples/conditioning.py
```

The example sends a precomputed `(..., 32)` latent directly through the
no-bridge route and maps a `(..., 22)` robot-action tensor through a synthetic
bridge. It produces model-ready `(..., 32)` tensors but does not invoke an
external ACWM.

Validate the non-null paths in a paper-budget config:

```bash
bash scripts/run.sh doctor \
  --strict \
  --config configs/pipeline_100h_2b.yaml
```

The configs intentionally leave model and dataset assets unset. The released
2B research checkpoints are downloaded separately and still require compatible
base models and data; these examples validate primitives and configuration
contracts, not the paper's trained metrics.
