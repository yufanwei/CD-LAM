# Checkpoints and bridge assets

## Availability

The compact tensor-exact 2B research snapshot is published at immutable
Hugging Face revision
`591e22e582e920cbb4fdfac1a45365e81088bd06` under
<https://huggingface.co/yufanwei/CD-LAM>. This source release does not contain
the large model files. The compact snapshot's `asset_manifest.json` declares
the authoritative filenames, roles, formats, byte sizes, SHA-256 values,
lineages, and compatibility metadata.

Consequently:

- unit tests and synthetic smoke tests are available now;
- configuration and asset validation are available now;
- the compact ACWM files are overlays and require their compatible external
  2B base model and user-obtained data; the source release supplies the pinned
  2B staging and launch wrappers;
- 14B execution and exact paper-metric reproduction require additional
  user-supplied or unreleased compatible assets;
- a random or similarly named checkpoint must never be substituted silently.

The compact manifest records for every file: role, model scale, training
stage, data tier, optimizer update, tensor format, byte size, SHA-256, public
runtime provenance, and base-model provenance. License and access terms are
recorded in the colocated model-license notice and external-dependency file.

## Logical checkpoint roles

| role | required by | compatibility requirement |
|---|---|---|
| base ACWM | Stages 2–3 | exact architecture, scale, latent dimension, and conditioning format |
| debiased LAM | Stages 2–3 | 32D encoder output and matching preprocessing |
| Stage-2 ACWM | Stage 3 | trained against the same debiased LAM space |
| 22D-to-32D bridge bundle | Stage 3 / rollout | matching LAM statistics and action convention |
| final Stage-3 ACWM | robot-action evaluation | matching scale, bridge contract, and action convention |

## Bridge contract

The Stage-3 bridge maps an AgiBot 22D recorded action into the 32D CD-LAM
space. A release artifact must include all of:

```text
g_state
action_mean
action_std
zm
zsd
latent_dim
```

The inference contract is:

```text
u_norm = (u - action_mean) / action_std
z_norm = g_state(u_norm)
z      = z_norm * zsd + zm
```

For the paper's AgiBot Alpha bridge, `u` is the stride-4 transition
`absolute_action[t + 4] - absolute_action[t]`. Its fixed ordering is arm 14,
grippers 2, head 2, waist 2, and robot velocity 2. The absolute vector comes
from the publisher's independent action-command arrays, not the observation
state. Shape validation cannot detect an absolute command, another cadence,
different units, a zero-filled velocity, or a reordered action. Those cases
require a bridge trained and calibrated for the new convention.

Loaders must validate finite statistics, nonzero standard deviations,
`action_dim=22`, and `latent_dim=32`. Bridge weights without their
`action_mean`, `action_std`, `zm`, and `zsd` are incomplete and should be
rejected.

The compact `action_contract.json` and bridge sidecars record the embodiment,
ordered action components, loader and bridge representations, source stride,
normalization-metadata hashes, and matching LAM checkpoint hash. Units and
coordinate frames remain inherited from the cited AgiBot modality metadata;
the converter, runtime doctor, and any custom backbone adapter must verify them
explicitly rather than infer them from shape or filename.

## Local routing

Copy the real-runtime template and set paths to assets you are authorized to
use:

```bash
cp configs/runtime.example.json configs/runtime.json
bash scripts/run.sh runtime-doctor --stage all
```

The template uses repository-relative defaults and contains no private machine
paths. Keep large data and model files outside Git; do not commit a populated
`runtime.json`. Set `CDLAM_RUNTIME_CONFIG=/absolute/path/to/runtime.json` when
the profile lives elsewhere. `configs/paths.local.env` remains available for
the lightweight planner commands, but it is not the real 2B runtime profile.
