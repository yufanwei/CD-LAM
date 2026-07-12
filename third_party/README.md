# External dependencies

CD-LAM does not vendor the complete ACWM backbone, SAM3, or CoWTracker. Their
pinned source revisions and license names are recorded in
[`dependencies.lock.json`](dependencies.lock.json).

The repository does include a sub-1-MiB, source-only ACWM integration overlay
under [`acwm_overlay/`](acwm_overlay/). It is hash-bound to the pinned NVIDIA
source revision and contains no model or dataset payloads.

The lock also records gated model assets used by the production stack. Entries
with `fetch_mode: manual` must be obtained from their model repositories after
accepting the applicable terms; `fetch_mode: source-script` entries are the
only ones handled by `scripts/fetch_optional_deps.sh`.

Fetch the metric backends only after reviewing and accepting their licenses:

```bash
CDLAM_ACCEPT_SAM3_LICENSE=yes \
CDLAM_ACCEPT_COWTRACKER_LICENSE=yes \
bash scripts/fetch_optional_deps.sh metrics
```

The sources are placed in `.deps/`, which is ignored by Git. In particular,
CoWTracker is distributed for noncommercial research use; the Apache-2.0
license of CD-LAM does not override any external license.

The fetch script applies a small H100/B200 and newer-`timm` compatibility patch
after checking out the pinned CoWTracker revision. That patch is a modification
of CoWTracker and remains subject to CoWTracker's license.

Fetch the pinned ACWM source and stage the verified CD-LAM runtime separately:

```bash
CDLAM_ACCEPT_BASE_LICENSE=yes bash scripts/fetch_optional_deps.sh base
```

The unmodified checkout is stored at `.deps/acwm-base-source`; the staged
runtime is stored at `.deps/acwm-runtime`. Re-running the command verifies the
existing runtime instead of silently overwriting it.

The baseline name is retained in the dependency URL and scientific results
for transparent attribution. It is not used as CD-LAM's package name or
runtime configuration namespace.
