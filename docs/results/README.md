# Paper result fixtures

[`paper_results.json`](paper_results.json) contains exact values transcribed
from manuscript Tables I–V together with protocol and comparability metadata.

These values are reference fixtures for documentation and validation. They are
not recomputed by this source checkout, and they must not be presented as a
fresh run. The released 2B research checkpoints are not asserted to reproduce
every headline table; full metric reproduction also requires the original
evaluation data and pinned metric dependencies.

Validate the JSON schema and paper-table invariants with:

```bash
bash run.sh validate-results
```

FDCE values from the zero-action static-reference protocol, Table-V ablation
protocol, ordinary rollout protocol, and target-transfer protocol are stored
with separate labels because their absolute values are not interchangeable.
