#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCEPTED="${CDLAM_ACCEPT_BASE_LICENSE:-}"
ARGS=()

usage() {
  cat <<'EOF'
Usage: bash setup.sh --accept-base-license [OPTIONS]

Create and validate the single CD-LAM GPU environment.

Options:
  --accept-base-license  Confirm that you reviewed the pinned upstream license.
  --gpu INDEX            GPU used for CUDA validation (default: 0).
  --with-models          Download and verify the three released models.
  --with-metrics         Fetch optional SAM3 and CoWTracker sources.
  --dry-run              Print setup operations without creating files.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-base-license)
      ACCEPTED=yes
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      ;;
  esac
  shift
done

if [[ "$ACCEPTED" != yes ]]; then
  echo "Review the pinned upstream license, then pass --accept-base-license." >&2
  exit 2
fi

export CDLAM_ACCEPT_BASE_LICENSE=yes
exec bash "$ROOT/scripts/bootstrap.sh" "${ARGS[@]}"
