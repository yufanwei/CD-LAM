# Evaluation protocol

The paper separates Stage-1 representation audits, Stage-2 latent-action
rollouts, and Stage-3 robot-action rollouts. Results from different protocols
must not be pooled.

> **Reproduction status:** the NumPy reducer in `cd_lam.metrics.fdce` implements
> the manuscript's fixed-track-pair Eq. A.3 followed by symmetric Chamfer Eq.
> A.4. The historical artifact pipeline used for the transcribed headline
> tables instead computed a Chamfer reduction independently at each frame and
> then averaged over time. These operations are not equivalent. A crossing
> trajectory fixture gives 5.0 with the manuscript reducer and approximately
> zero with the historical order. The released table values are therefore not
> canonical-code recomputations until the frozen populations are rescored with
> this module.

The release does not yet contain the exact 300-clip Stage-2 and Stage-3
population manifests, donor mapping, or a complete rollout-to-SAM3-to-
CoWTracker runner. `configs/eval_paper.yaml` is a normative protocol template,
not a frozen evaluation index. Comparable results must record episode and
window IDs, timestamps, donor IDs, seeds, crop and resolution, dependency
revisions, expected count, valid count, and every failure.

Once protocol-compatible tracks exist, the reducer is executable rather than
descriptive. Run `bash run.sh score-fdce` with `--tracks` and
`--output`; it accepts only 49-frame NPZ bundles with at most 16 anchors per
side by default, hashes every input, and writes per-sample plus mean/median
results. See [Evaluation](EVALUATION.md). This closes metric reduction, not the
missing frozen-population or rollout-generation boundary.

## Foreground Displacement Chamfer Error (FDCE)

FDCE measures action following from foreground displacement tracks rather than
raw appearance. SAM3 selects embodiment and interacted-object regions;
CoWTracker produces point tracks only in valid foreground regions.

For reference point `p_j^s` and generated point `p_hat_i^s` at rollout step
`s`, define displacement relative to each track's initial point:

```text
a_j^s     = p_j^s     - p_j^0
a_hat_i^s = p_hat_i^s - p_hat_i^0

c_ij = (1/H) sum_{s=1..H} ||a_hat_i^s - a_j^s||_2
```

For `N_g` generated tracks and `N_r` reference tracks, the paper uses the
**symmetric Chamfer distance**

```text
FDCE(o_hat, o)
  = (1 / (2 N_g)) sum_{i=1..N_g} min_j c_ij
  + (1 / (2 N_r)) sum_{j=1..N_r} min_i c_ij.
```

The protocol samples up to 16 valid foreground anchors per rollout pair. Seeds
come from an eroded foreground mask, low-visibility tracks are discarded, and
distances are reported in pixels at evaluation resolution. Lower is better.
Mean FDCE is sensitive to catastrophic failures; median FDCE describes typical
behavior.

SAM3 is used for both FDCE foreground masks and the paper's Stage-1 foreground
weighting. CoWTracker is used for FDCE only, never for training. Both remain
optional external source/cache tools, not additional CD-LAM environments. Both
(or protocol-compatible cached outputs) are required for exact FDCE
reproduction.

## Stage-1 LAM audit (Table I)

Table I evaluates the encoder alone, before any world-model rollout. Lower is
better for every reported diagnostic.

| diagnostic | baseline LAM | CD-LAM |
|---|---:|---:|
| zero-transition response, median relative norm | 0.527 | 0.043 |
| absolute latent norm, median | 3.119 | 0.226 |
| horizontal camera shift, mean / median relative norm | 0.555 / 0.536 | 0.156 / 0.096 |
| vertical camera shift, mean / median relative norm | 0.545 / 0.529 | 0.110 / 0.064 |
| shortcut leakage | 0.151 | 0.014 |

The preservation check reported in the text is the cosine similarity of
same-primitive pairs from different episodes: 0.132 for the baseline and 0.131
for CD-LAM. It supports the interpretation that the diagnostic improvements
are not explained by uniformly shrinking local action structure.

## Stage-2 latent-action rollouts (Table II)

Evaluation uses 300 held-out EgoDex clips. The left block uses latent actions
extracted from the video's own transition. The right block applies target
latent transfer `do(z_t = z_t^tar)` under fixed source context. Pixel-metric
differences in the transfer block are marginal; FDCE is the operative transfer
comparison.

| backbone | model | own-z FDCE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | transfer PSNR ↑ | transfer SSIM ↑ | transfer LPIPS ↓ | transfer FDCE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B | DreamDojo | 34.00 | 20.88 | 0.780 | 0.413 | 13.02 | 0.598 | 0.643 | 42.74 |
| 2B | CD-LAM | **19.63** | **24.29** | **0.827** | **0.308** | **13.15** | 0.588 | **0.600** | **33.81** |
| 14B | DreamDojo | 40.29 | 21.04 | 0.792 | 0.398 | **13.11** | 0.593 | 0.631 | 50.27 |
| 14B | CD-LAM | **29.87** | **23.18** | **0.814** | **0.342** | 13.03 | **0.597** | **0.617** | **33.22** |

Own-latent FDCE falls by 42% at 2B and 26% at 14B (rounded as in the
manuscript). Transfer FDCE falls by 21% and 34%, respectively.

## Stage-3 robot-action rollouts (Table III)

Evaluation uses 300 AgiBot clips drawn from distinct episodes. The normal
rollout consumes the recorded robot action. `do(u_t=0)` measures residual
motion against a static reference made by holding the initial frame fixed.
That zero-action FDCE is therefore **not numerically comparable** with the
normal or transfer rollout columns. `do(u_t=u_t^tar)` tests target-action
transfer under fixed source context.

| backbone | model | rollout FDCE mean ↓ | FDCE median ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | zero-action FDCE ↓ | transfer FDCE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2B | DreamDojo | 12.63 | 8.15 | 19.85 | 0.798 | 0.271 | 10.71 | 24.36 |
| 2B | CD-LAM | **8.24** | **6.75** | **20.60** | **0.806** | **0.269** | **5.03** | **22.55** |
| 14B | DreamDojo | 11.11 | 8.98 | 20.01 | 0.808 | 0.263 | 9.36 | 24.82 |
| 14B | CD-LAM | **7.73** | **5.99** | **21.01** | **0.818** | **0.247** | **2.18** | **21.11** |

FDCE mean falls by 35% at 2B and 30% at 14B (rounded as in the manuscript).
The 14B CD-LAM row is better than its 2B counterpart in every Table-III
column. These are action-following and video-fidelity results; they do not
establish planning, policy, or real-robot task-success performance.

## Other image metrics

- PSNR and SSIM: higher is better.
- LPIPS: lower is better.
- FG-PSNR: PSNR restricted to the foreground mask; higher is better.

Pixel metrics are computed against the reference rollout on full frames unless
explicitly marked foreground-only. A high PSNR does not by itself certify that
the commanded foreground motion was followed.

## Known limitations

Heavy hand-object occlusion, motion blur, textureless grippers, segmentation
errors, and tracker drift can inflate FDCE. Evaluation should record dependency
versions, mask/track validity counts, image resolution, seed, and aggregation
logic. Exact paper reproduction additionally requires a checkpoint explicitly
verified for the corresponding table and the original evaluation assets. The
released 2B research files do not by themselves satisfy that requirement.
Historical Beta results decoded through the old `torchvision_av` fallback are
invalid because strided timestamp requests collapsed to consecutive frames.
The source validator and pinned patch reject that implementation. At 30 FPS,
50 stride-four requests must span about 6.533 seconds, not 1.633 seconds.
