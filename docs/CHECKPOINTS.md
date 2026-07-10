# Checkpoints and bridge assets

## Availability

Public CD-LAM checkpoints and associated Hugging Face assets are **pending
upload** at <https://huggingface.co/yufanwei/CD-LAM>. This source release does
not contain trained model files, and it does not declare checkpoint filenames
or checksums before the uploads exist.

Consequently:

- unit tests and synthetic smoke tests are available now;
- configuration and asset validation are available now;
- full Stage-1/2/3 training and paper-metric reproduction require user-supplied
  compatible backbones, data, and/or the pending release assets;
- a random or similarly named checkpoint must never be substituted silently.

When assets are published, a versioned manifest should record for every file:
role, model scale, training stage, data tier, optimizer update, tensor format,
byte size, SHA-256, license/access terms, and the exact code revision.

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
grippers 2, head 2, waist 2, and base 2. Shape validation cannot detect an
absolute command, another cadence, different units, or a reordered action.
Those cases require a bridge trained and calibrated for the new convention.

Loaders must validate finite statistics, nonzero standard deviations,
`action_dim=22`, and `latent_dim=32`. Bridge weights without their
`action_mean`, `action_std`, `zm`, and `zsd` are incomplete and should be
rejected.

Before public distribution, a bridge manifest should additionally record the
robot/dataset identity, ordered action components, units, coordinate frames,
`action_representation=stride_delta`, source stride, upstream normalization
convention, preprocessing revision, and matching LAM checkpoint hash. These
semantic fields are required for safe reuse even when older research bundles
do not contain them.

## Local routing

Copy the portable template and set paths to assets you are authorized to use:

```bash
cp configs/paths.example.env configs/paths.local.env
```

The template uses repository-relative defaults and contains no private machine
paths. Keep large data and model files outside Git; do not commit a populated
`paths.local.env`. `scripts/run.sh` loads this profile automatically and maps
its exported asset paths into the typed pipeline configuration.
