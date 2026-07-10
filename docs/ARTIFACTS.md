# Release artifacts

## Source repository

<https://github.com/yufanwei/CD-LAM> contains the public source, portable
configs, tests, documentation, and machine-readable paper-result fixtures.

## Project page

<https://yufanwei.github.io/CD-LAM-project-page/> is the public project page.

## Model repository

<https://huggingface.co/yufanwei/CD-LAM> contains the tensor-exact 2B research
checkpoint release. The 100h and 1000h lineages each include a Stage-1 LAM,
Stage-2 and Stage-3 ACWM overlays, and a matching bridge. No trained checkpoint
is bundled with this source tree; `asset_manifest.json` in the model repository
is authoritative for filenames, roles, sizes, SHA-256 values, and lineage.

`bash scripts/run.sh download-models` fails closed until the snapshot contains
a tensor-exact `asset_manifest.json` with at least one released asset. Selective
`--allow-pattern` downloads always include that manifest and must match a
declared asset path; downloaded files are checked against their recorded size
and SHA-256.

## Included versus external

| artifact | source release | external availability |
|---|---:|---:|
| loss, action-transform, bridge, and evaluation primitives | yes | no |
| typed stage runners and synthetic optimizer/checkpoint smoke | yes | no |
| portable protocol configs, adapter interface, and asset validation | yes | no |
| exact paper tables as JSON | yes | no |
| 2B CD-LAM research checkpoints | no | released on Hugging Face |
| 14B CD-LAM checkpoints | no | not released |
| base 2B/14B backbones | no | user-supplied under upstream terms |
| Cosmos video tokenizer and text encoder | no | pinned gated NVIDIA assets |
| EgoDex and AgiBot data | no | obtain from dataset owners |
| SAM3 source and weights | no | optional external dependency |
| CoWTracker source and weights | no | optional external dependency |
| paper rollout videos and metric caches | no | not released |

SAM3 is used for Stage-1 foreground masks and FDCE masks. CoWTracker is used
only for FDCE point tracks. Neither is required by the synthetic core smoke
tests, and neither is covered by the CD-LAM repository license. Review and
accept each upstream license and weight-access policy before installation.

## Reproducibility boundary

The current quickstart checks code primitives, action algebra, bridge
serialization, typed plans, real CPU optimizer/checkpoint/resume paths, and
dependency gates. Downloaded release files are additionally checked against
the tensor-exact manifest. End-to-end paper reproduction still requires the
compatible base checkpoint, datasets, a pinned production adapter and GPU
environment, and all training correctness gates; the model card explicitly
qualifies which released research assets are not verified headline-table
reproductions.
