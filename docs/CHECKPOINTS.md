# Checkpoints and bridge assets

## Availability

Tensor-exact 2B research checkpoints are published at
<https://huggingface.co/yufanwei/CD-LAM>. This source release does not contain
the large model files. The model repository's `asset_manifest.json` declares
the authoritative filenames, roles, formats, byte sizes, SHA-256 values,
lineages, and compatibility metadata.

Consequently:

- unit tests and synthetic smoke tests are available now;
- configuration and asset validation are available now;
- the released ACWM files are overlays and require their compatible external
  2B base model, production adapter, and data;
- 14B execution and exact paper-metric reproduction require additional
  user-supplied or unreleased compatible assets;
- a random or similarly named checkpoint must never be substituted silently.

The published manifest records for every file: role, model scale, training
stage, data tier, optimizer update, tensor format, byte size, SHA-256,
license/access terms, and source-code provenance.

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

The published `action_contract.json` and bridge sidecars record the embodiment,
ordered action components, loader and bridge representations, source stride,
normalization-metadata hashes, and matching LAM checkpoint hash. Units and
coordinate frames remain inherited from the cited AgiBot modality metadata;
a production adapter must verify them explicitly rather than infer them from
shape or filename.

## Local routing

Copy the portable template and set paths to assets you are authorized to use:

```bash
cp configs/paths.example.env configs/paths.local.env
```

The template uses repository-relative defaults and contains no private machine
paths. Keep large data and model files outside Git; do not commit a populated
`paths.local.env`. `scripts/run.sh` loads this profile automatically and maps
its exported asset paths into the typed pipeline configuration.
