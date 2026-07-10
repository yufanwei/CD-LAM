#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATHS_ENV="${CDLAM_PATHS_ENV:-$ROOT/configs/paths.local.env}"
if [[ -f "$PATHS_ENV" ]]; then
  export CDLAM_ROOT="${CDLAM_ROOT:-$ROOT}"
  # This is an explicit, user-owned shell profile. The example exports only
  # CD-LAM paths and executable selection.
  source "$PATHS_ENV"
fi
if [[ -x "${CDLAM_VENV:-$ROOT/.venv}/bin/python" ]]; then
  PY="${CDLAM_VENV:-$ROOT/.venv}/bin/python"
else
  PY="${PYTHON:-python3}"
fi

run_lint() {
  local venv_ruff="${CDLAM_VENV:-$ROOT/.venv}/bin/ruff"
  if [[ -x "${CDLAM_RUFF_BIN:-$venv_ruff}" ]]; then
    "${CDLAM_RUFF_BIN:-$venv_ruff}" check "$ROOT" "$@"
  elif command -v ruff >/dev/null 2>&1; then
    ruff check "$ROOT" "$@"
  else
    "$PY" -m ruff check "$ROOT" "$@"
  fi
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run.sh COMMAND [ARGS...]

Commands:
  doctor             Check the install and optional/full-reproduction assets.
  smoke              Run deterministic bridge/objective/FDCE smoke checks.
  train-smoke        Run Stage 1, Stage 2, bridge, and Stage 3 CPU training smoke.
  data-prepare       Build all staged manifests from episode JSONL metadata.
  data-validate      Validate staged manifest schemas and alignment.
  stage1             Plan or run Stage-1 training.
  stage2             Plan or run Stage-2 training.
  bridge-train       Plan or run action-to-latent bridge training.
  stage3             Plan or run Stage-3 training.
  test               Run the unit-test suite.
  lint               Run Ruff source checks.
  check              Run all CPU source-release gates.
  validate-results   Validate the machine-readable paper tables.
  fetch-metrics      Fetch pinned SAM3 and CoWTracker sources.
  fetch-base         Fetch the pinned external ACWM source.
  fetch-all          Fetch the base and metric sources.
  download-models    Download released artifacts from Hugging Face.
  release-check      Check paths, permissions, JSON, shell, and package hygiene.
EOF
}

command="${1:-}"
[[ -n "$command" ]] || { usage >&2; exit 2; }
shift

case "$command" in
  doctor) "$PY" -m cd_lam doctor "$@" ;;
  smoke) "$PY" -m cd_lam smoke "$@" ;;
  train-smoke) "$PY" -m cd_lam train-smoke "$@" ;;
  data-prepare) "$PY" -m cd_lam data-prepare "$@" ;;
  data-validate) "$PY" -m cd_lam data-validate "$@" ;;
  stage1|stage2|bridge-train|stage3) "$PY" -m cd_lam "$command" "$@" ;;
  test) "$PY" -m pytest -q "$ROOT/tests" "$@" ;;
  lint) run_lint "$@" ;;
  check) make -C "$ROOT" PYTHON="$PY" check "$@" ;;
  validate-results) "$PY" "$ROOT/tools/validate_results.py" "$@" ;;
  fetch-metrics) bash "$ROOT/scripts/fetch_optional_deps.sh" metrics "$@" ;;
  fetch-base) bash "$ROOT/scripts/fetch_optional_deps.sh" base "$@" ;;
  fetch-all) bash "$ROOT/scripts/fetch_optional_deps.sh" all "$@" ;;
  download-models) "$PY" "$ROOT/scripts/download_models.py" "$@" ;;
  release-check) "$PY" "$ROOT/tools/release_check.py" --strict "$@" ;;
  -h|--help|help) usage ;;
  *) echo "Unknown command: $command" >&2; usage >&2; exit 2 ;;
esac
