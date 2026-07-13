# Release artifacts

## Source repository

<https://github.com/yufanwei/CD-LAM> contains the public source, portable
configs, tests, documentation, and machine-readable paper-result fixtures.

## Project page

<https://yufanwei.github.io/CD-LAM-project-page/> is the public project page.

## Model repository

The compact snapshot at
<https://huggingface.co/yufanwei/CD-LAM> contains exactly three main
tensor-exact 2B research entries. Immutable revision
`2a02c07f4f5d1b731e0cf4e4fa1b9767ed592c1f` retains the tensor-verified model
blobs and passed the guarded 10-file inventory, manifest, size, and LFS
SHA-256 checks:

| entry | path | role |
|---|---|---|
| LAM | `models/lam/model.pt` | selected masked 32D latent-action model |
| pretrain | `models/pretrain/model.pt` | pretrained ACWM overlay compatible with the compact LAM |
| posttrain | `models/posttrain/model.pt` | selected robot-action posttraining overlay |

`models/posttrain/` also contains `bridge.pt` and
`action_contract.json`. They are compatibility auxiliaries for the posttrained
entry, not additional main models. No trained checkpoint is bundled with this
source tree; `asset_manifest.json` in the model repository is authoritative
for filenames, roles, sizes, SHA-256 values, and lineage.

The LAM and pretrain entries form one compatible pair. The posttrain entry is
bound to the colocated bridge.
Do not load all three as a sequential pipeline or substitute the public LAM
for the posttrain entry's recorded latent-space identity.

`bash run.sh download-models` selects that immutable revision by default
and fails closed unless the snapshot contains release ID
`cd-lam-2b-three-entry`, the exact three model identities, the two required
posttraining auxiliaries, a pinned public runtime, a pinned base model, and a
tensor-exact manifest. Selective `--allow-pattern` downloads always include
that manifest and must match a declared asset path; downloaded files are
checked against their recorded size and SHA-256.

## Included versus external

| artifact | source release | external availability |
|---|---:|---:|
| loss, action-transform, bridge, and evaluation primitives | yes | no |
| typed stage planners and synthetic optimizer/checkpoint smoke | yes | no |
| pinned 2B launch wrappers and manifest-checked source overlay | yes | complete upstream source is staged outside Git |
| portable protocol configs, custom-backbone interface, and asset validation | yes | no |
| exact paper tables as JSON | yes | no |
| three main 2B CD-LAM research entries | no | published at immutable revision `2a02c07` |
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
serialization, typed plans, deterministic checkpoint/resume paths, a live CUDA
optimizer step, and dependency gates. Downloaded release files are additionally checked against
the tensor-exact manifest. End-to-end paper reproduction still requires the
compatible base checkpoint, datasets, the staged pinned 2B runtime, the
supported `.venv` and GPU host, and all training correctness gates; the model card explicitly
qualifies which released research assets are not verified headline-table
reproductions.
