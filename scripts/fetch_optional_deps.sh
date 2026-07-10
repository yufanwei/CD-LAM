#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${CDLAM_DEPS_DIR:-$ROOT/.deps}"

SAM3_URL="https://github.com/facebookresearch/sam3.git"
SAM3_REV="8e451d5eb43c817b64ae7577fb7b9ae223db88a9"
COWTRACKER_URL="https://github.com/facebookresearch/cowtracker.git"
COWTRACKER_REV="1454f20045d3b514e5b8417907152677f3dba621"
BASE_URL="https://github.com/NVIDIA/DreamDojo.git"
BASE_REV="02f119b759d5c7f84a399fdeea3c6e82e7ed6cff"

usage() {
  cat <<'EOF'
Usage: bash scripts/fetch_optional_deps.sh metrics|base|all

License acceptance is explicit:
  metrics: CDLAM_ACCEPT_SAM3_LICENSE=yes and
           CDLAM_ACCEPT_COWTRACKER_LICENSE=yes
  base:    CDLAM_ACCEPT_BASE_LICENSE=yes

Sources are cloned at revisions recorded in third_party/dependencies.lock.json.
Nothing is vendored into the CD-LAM Git history.
EOF
}

clone_at() {
  local name="$1" url="$2" rev="$3" dest="$4"
  if [[ -e "$dest" && ! -d "$dest/.git" ]]; then
    echo "Refusing to overwrite non-git path: $dest" >&2
    exit 2
  fi
  if [[ ! -d "$dest/.git" ]]; then
    mkdir -p "$(dirname "$dest")"
    git clone --filter=blob:none "$url" "$dest"
  fi
  local origin
  origin="$(git -C "$dest" remote get-url origin)"
  [[ "$origin" == "$url" ]] || {
    echo "$name origin mismatch: expected $url, got $origin" >&2
    exit 2
  }
  git -C "$dest" fetch --depth 1 origin "$rev"
  git -C "$dest" checkout --detach "$rev"
  local actual
  actual="$(git -C "$dest" rev-parse HEAD)"
  [[ "$actual" == "$rev" ]] || {
    echo "$name revision mismatch: expected $rev, got $actual" >&2
    exit 2
  }
  echo "OK $name $actual"
}

target="${1:-}"
case "$target" in
  metrics)
    [[ "${CDLAM_ACCEPT_SAM3_LICENSE:-}" == yes ]] || {
      echo "Read the SAM3 license, then set CDLAM_ACCEPT_SAM3_LICENSE=yes." >&2
      exit 2
    }
    [[ "${CDLAM_ACCEPT_COWTRACKER_LICENSE:-}" == yes ]] || {
      echo "Read the CoWTracker noncommercial research license, then set CDLAM_ACCEPT_COWTRACKER_LICENSE=yes." >&2
      exit 2
    }
    clone_at SAM3 "$SAM3_URL" "$SAM3_REV" "$DEST_ROOT/sam3"
    clone_at CoWTracker "$COWTRACKER_URL" "$COWTRACKER_REV" "$DEST_ROOT/cowtracker"
    git -C "$DEST_ROOT/cowtracker" submodule update --init --recursive
    patch_file="$ROOT/third_party/patches/cowtracker_flash_attention_compat.patch"
    if git -C "$DEST_ROOT/cowtracker" apply --check "$patch_file"; then
      git -C "$DEST_ROOT/cowtracker" apply "$patch_file"
      echo "OK CoWTracker compatibility patch applied"
    elif git -C "$DEST_ROOT/cowtracker" apply --reverse --check "$patch_file"; then
      echo "OK CoWTracker compatibility patch already applied"
    else
      echo "CoWTracker tree does not match the pinned compatibility patch" >&2
      exit 2
    fi
    ;;
  base)
    [[ "${CDLAM_ACCEPT_BASE_LICENSE:-}" == yes ]] || {
      echo "Read the base ACWM license, then set CDLAM_ACCEPT_BASE_LICENSE=yes." >&2
      exit 2
    }
    clone_at ACWM-base "$BASE_URL" "$BASE_REV" "$DEST_ROOT/acwm-base"
    ;;
  all)
    "$0" metrics
    "$0" base
    ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
