# CD-LAM training pipeline

CD-LAM is a three-stage fine-tuning pipeline. Stages 1 and 2 use
action-unlabeled video; Stage 3 introduces paired robot actions. The main paper
setting uses the 100-hour debiasing tier.

## Stage 1: LAM debiased fine-tuning

For a transition `(o_t, o_{t+1})`, the encoder produces a 32D latent action
`z_t = mu_phi(o_t, o_{t+1})`. Stage 1 optimizes three complementary terms:

```text
L_CD = L_emb + lambda_ctr(k) L_ctr + lambda_cal L_cal
L_cal = L_KL-fb + L_zero
```

- **Embodiment-centric reconstruction (`L_emb`)** weights the embodiment and
  interacted-object foreground more strongly than the background. The paper
  obtains the soft foreground mask with SAM3 while retaining a nonzero
  background weight.
- **Action-centric contrast (`L_ctr`)** pulls transitions with the same coarse
  action primitive together across visual contexts and pushes different
  primitives apart. The labels are 12-way caption-derived verb categories, not
  executable robot controls.
- **Latent-space calibration (`L_cal`)** combines a free-bit KL term with a
  zero-transition term. The latter anchors duplicated-frame inputs
  `(o_t, o_t)` near a zero-transition reference without collapsing ordinary
  transitions.

Paper budget: **1,000 optimizer updates**, per-GPU batch size 32, with 100 hours
of video for the main result. Table IV varies only the data tier (1h, 10h,
100h, 1000h) while keeping this 1,000-update budget fixed.

SAM3 is an optional software dependency for the released primitives, but its
masks (or protocol-compatible precomputed masks) are required to reproduce the
paper's embodiment-centric weighting exactly.

## Stage 2: ACWM debiased fine-tuning

The fine-tuned LAM extracts debiased actions `z_t^CD` from video transitions.
The ACWM is then continued with these latents in the existing action
conditioning format. There is no executable robot-action bridge in Stage 2.

Paper budget: **2,000 optimizer updates**; per-GPU batch size 12 for 2B and 2
for 14B.

This stage is evaluated on 300 held-out EgoDex clips using ordinary
latent-action rollouts and target-latent transfer under fixed source context.

## Stage 3: robot-action adaptation

For paired `(o_t, u_t, o_{t+1})` data, a lightweight bridge maps an aligned
**22D robot-action transition** `u_t` to the **32D debiased latent action**.
In the paper's AgiBot Alpha pipeline, this input is
`absolute_action[t + 4] - absolute_action[t]` with ordering arm 14, grippers 2,
head 2, waist 2, and base 2:

```text
u_norm = (u_t - action_mean) / action_std
z_norm = g_eta(u_norm)
z_hat  = z_norm * zsd + zm
```

The bridge is trained to regress the gradient-stopped CD-LAM encoder target,
with the paper's auxiliary cycle/readout terms preserving decodability of the
recorded action. The ACWM consumes `z_hat` through the same latent action
conditioning path used in Stage 2.

A usable bridge artifact is a bundle, not just an MLP state. It must preserve
the learned parameters and normalization contract:

```text
g_state, action_mean, action_std, zm, zsd, latent_dim
```

Expected dimensions are `action_dim=22` and `latent_dim=32`. Loaders should
reject missing statistics or mismatched shapes.

Final paper checkpoints: **3,000 optimizer updates for 2B** and **6,000 for
14B**. Stage 3 is evaluated on 300 AgiBot clips drawn from distinct episodes.

## Main training budgets

| stage | main data | 2B updates | 14B updates | paper per-GPU batch size |
|---|---|---:|---:|---:|
| 1. LAM debiased fine-tuning | 100h action-unlabeled video | 1,000 | 1,000 | 32 |
| 2. ACWM debiased fine-tuning | action-unlabeled video | 2,000 | 2,000 | 12 / 2 |
| 3. robot-action adaptation | paired robot action video | 3,000 | 6,000 | not specified in the manuscript |

The manuscript reports that training used 96 H100 GPUs. The public configs
record optimizer-update budgets, not a promise of identical wall-clock time on
other hardware.

## Validating a release configuration

The portable protocol configs are
[`pipeline_100h_2b.yaml`](../configs/pipeline_100h_2b.yaml) and
[`pipeline_100h_14b.yaml`](../configs/pipeline_100h_14b.yaml). After filling
their required asset fields locally, validate every configured path with:

```bash
bash scripts/run.sh doctor \
  --strict \
  --config configs/pipeline_100h_2b.yaml
```

Public trained assets are pending upload. The current release provides typed
stage commands, an optimizer-based CPU integration backend, and a production
adapter interface. Real adapters fail closed while required assets are absent;
the repository does not claim end-to-end paper reproduction from a source-only
clone. See [`TRAINING.md`](TRAINING.md).
