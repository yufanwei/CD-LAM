# Release artifacts

## Source repository

<https://github.com/yufanwei/CD-LAM> contains the public source, portable
configs, tests, documentation, and machine-readable paper-result fixtures.

## Project page

<https://yufanwei.github.io/CD-LAM-project-page/> is the public project page.

## Model repository

<https://huggingface.co/yufanwei/CD-LAM> is the declared location for model and
large evaluation assets. **Uploads are pending.** No trained checkpoint is
bundled with this source tree, and no filename or checksum is promised until a
versioned asset manifest is published there.

`bash scripts/run.sh download-models` fails closed until the snapshot contains
a tensor-exact `asset_manifest.json` with at least one released asset. Selective
`--allow-pattern` downloads always include that manifest and must match a
declared asset path; downloaded files are checked against their recorded size
and SHA-256.

## Included versus external

| artifact | source release | external/pending |
|---|---:|---:|
| loss, action-transform, bridge, and evaluation primitives | yes | no |
| typed stage runners and synthetic optimizer/checkpoint smoke | yes | no |
| portable protocol configs, adapter interface, and asset validation | yes | no |
| exact paper tables as JSON | yes | no |
| CD-LAM trained checkpoints | no | pending on Hugging Face |
| base 2B/14B backbones | no | user-supplied under upstream terms |
| Cosmos video tokenizer and text encoder | no | pinned gated NVIDIA assets |
| EgoDex and AgiBot data | no | obtain from dataset owners |
| SAM3 source and weights | no | optional external dependency |
| CoWTracker source and weights | no | optional external dependency |
| paper rollout videos and metric caches | no | pending/when released |

SAM3 is used for Stage-1 foreground masks and FDCE masks. CoWTracker is used
only for FDCE point tracks. Neither is required by the synthetic core smoke
tests, and neither is covered by the CD-LAM repository license. Review and
accept each upstream license and weight-access policy before installation.

## Reproducibility boundary

The current quickstart checks code primitives, action algebra, bridge
serialization, typed plans, real CPU optimizer/checkpoint/resume paths, and
dependency gates. It does not recreate the paper's trained models. End-to-end
reproduction additionally requires compatible checkpoints, datasets, a pinned
production adapter and GPU environment, and all training correctness gates.
