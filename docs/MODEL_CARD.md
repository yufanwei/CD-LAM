---
license: other
library_name: pytorch
tags:
  - latent-action-model
  - world-model
  - robotics
  - video-prediction
  - research
---

# CD-LAM model card

## Model summary

CD-LAM (Causally Debiased Latent Action Model) is a method for improving the
latent action condition used by embodied action conditioned world models. It
adds three LAM fine-tuning objectives, adapts the ACWM to the repaired latent
space, and then learns a 22D-robot-action-to-32D-latent bridge.

This card describes the method and release family. Tensor-exact 2B research
checkpoints for the 100h and 1000h lineages are published at
<https://huggingface.co/yufanwei/CD-LAM>. They include Stage-1 LAM, Stage-2 and
Stage-3 ACWM overlays, and matching 22D-to-32D bridges. The source checkout
provides typed plans and method primitives; applying an overlay still requires
the compatible external base model and production adapter.

## Architecture and training stages

1. LAM debiased fine-tuning with embodiment-centric reconstruction,
   action-centric contrast, and latent-space calibration.
2. ACWM debiased fine-tuning using the resulting 32D latent actions.
3. Robot-action adaptation using a normalized 22D-to-32D MLP bridge and the
   same ACWM action-conditioning interface.

The main manuscript setting uses 100 hours of debiasing video, 1,000 Stage-1
updates, 2,000 Stage-2 updates, and final Stage-3 checkpoints at 3,000 updates
for 2B and 6,000 for 14B.

## Data

- Stages 1–2 use action-unlabeled video; Stage-1 contrast labels are coarse
  12-way caption-derived verb categories, not executable robot actions.
- Stage 3 uses paired AgiBot video and stride-4 action transitions. The paper
  bridge input is `absolute_action[t + 4] - absolute_action[t]` in the fixed
  arm-14, grippers-2, head-2, waist-2, base-2 order.
- Stage-2 evaluation uses 300 held-out EgoDex clips.
- Stage-3 evaluation uses 300 AgiBot clips from distinct episodes.

Datasets are not redistributed by this repository. Users are responsible for
obtaining them and complying with their licenses and terms.

## Reported evaluation

The principal Stage-3 manuscript results are:

| backbone | model | FDCE mean ↓ | FDCE median ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| 2B | DreamDojo | 12.63 | 8.15 | 19.85 | 0.798 | 0.271 |
| 2B | CD-LAM | 8.24 | 6.75 | 20.60 | 0.806 | 0.269 |
| 14B | DreamDojo | 11.11 | 8.98 | 20.01 | 0.808 | 0.263 |
| 14B | CD-LAM | 7.73 | 5.99 | 21.01 | 0.818 | 0.247 |

These are manuscript values, not results recomputed by this source release.
See [`results/paper_results.json`](results/paper_results.json) for all Tables
I–V and [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) for intervention and
comparability details.

## Intended use

- Research on latent-action representation bias.
- Research on action-conditioned video/world models.
- Reproduction and extension of the documented Stage-1/2/3 protocols once
  compatible assets are available.
- Offline action-following and visual-fidelity evaluation.

## Out-of-scope use

CD-LAM has not been validated as a robot planner, policy, safety controller,
or autonomous deployment system. The reported results do not measure task
success, collision avoidance, robustness to open-world hazards, or human
safety. Do not use generated video as the sole basis for physical actuation or
high-stakes decisions.

## Limitations

- The released files cover two 2B research lineages, not a verified 14B or
  complete headline-table reproduction; the compatible base model, original
  data, and evaluation assets remain separate requirements.
- Results cover the datasets, backbones, action conventions, and scales in the
  manuscript; generalization beyond them is not established.
- FDCE depends on segmentation and tracking. Occlusion, motion blur,
  textureless grippers, mask errors, and tracker drift can inflate it.
- The 22D action layout, units, stride/delta convention, matching LAM space,
  and bridge normalization statistics are one compatibility contract and are
  not interchangeable across embodiments. Older research checkpoints may
  require this semantic metadata to be supplied separately.
- The reference loader's normalized block-anchor deltas must be converted to
  raw adjacent deltas before the paper bridge. The public action utility tests
  this conversion; external adapters remain responsible for wiring it before
  reduced-precision casting.
- The “more than 12x” result compares aligned optimizer updates needed to reach
  a baseline metric reference. It is not a wall-clock or final-checkpoint
  speedup claim.

## External dependencies and licenses

Apache-2.0 covers the CD-LAM source code in this repository. Any fine-tuned
weights published for CD-LAM also inherit the applicable NVIDIA base-model
terms; training datasets and optional metric dependencies retain their own
licenses and access conditions. The model repository therefore uses
`license: other` instead of presenting every artifact as Apache-2.0.

SAM3 is used to produce the paper's Stage-1 and FDCE foreground masks.
CoWTracker is used only for FDCE tracks. They are not vendored and remain under
their own source and weight licenses; the CD-LAM Apache-2.0 license does not
override those terms. See [`../third_party/README.md`](../third_party/README.md).

## Citation

See [`../CITATION.cff`](../CITATION.cff). No DOI or arXiv identifier is asserted by
this release until a versioned identifier is published by the authors.
