#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/configs/model_runtime.lock.json"
EXTRA_LOCK="$ROOT/configs/model_runtime.extra.lock.txt"
DEPS_ROOT="$ROOT/.deps"
MODEL_ENV="${CDLAM_VENV:-$ROOT/.venv}"
PYTHON="${CDLAM_MODEL_BOOTSTRAP_PYTHON:-python3.10}"
DRY_RUN=no

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_model_runtime.sh [OPTIONS]

Low-level helper used by scripts/bootstrap.sh to create or verify CD-LAM's
single Linux x86-64 CUDA 12.8 environment. The command fetches pinned source
and Python packages only; it never downloads model weights, datasets, tokenizer
files, or text encoders.

Options:
  --environment PATH  Target environment (default: .venv).
  --deps-root PATH    Staged-source root (default: .deps).
  --python PATH       CPython 3.10 bootstrap interpreter.
  --dry-run           Print the setup operations without creating files.
  -h, --help          Show this help.

When the staged source is absent, first review its license and set
CDLAM_ACCEPT_BASE_LICENSE=yes. An existing target environment is never changed;
it is validated and returned as-is.
EOF
}

fail() {
  echo "bootstrap_model_runtime: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --environment)
      [[ $# -ge 2 ]] || fail "--environment requires a path"
      MODEL_ENV="$2"
      shift 2
      ;;
    --deps-root)
      [[ $# -ge 2 ]] || fail "--deps-root requires a path"
      DEPS_ROOT="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || fail "--python requires a path"
      PYTHON="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=yes
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) fail "unknown option: $1" ;;
  esac
done

if [[ "$PYTHON" == */* ]]; then
  [[ -x "$PYTHON" ]] || fail "Python is not executable: $PYTHON"
else
  PYTHON="$(command -v "$PYTHON" || true)"
  [[ -n "$PYTHON" ]] || fail "CPython 3.10 was not found; pass --python PATH"
fi

"$PYTHON" - <<'PY' || fail "requires CPython 3.10 on Linux x86-64 with glibc >=2.35"
import platform
import sys

def parts(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(".") if part.isdigit())

assert platform.python_implementation() == "CPython"
assert sys.version_info[:2] == (3, 10)
assert platform.system() == "Linux"
assert platform.machine() == "x86_64"
assert parts(platform.libc_ver()[1]) >= (2, 35)
PY
absolute_path() {
  "$PYTHON" - "$1" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

DEPS_ROOT="$(absolute_path "$DEPS_ROOT")"
MODEL_ENV="$(absolute_path "$MODEL_ENV")"
ACWM_ROOT="$DEPS_ROOT/acwm-runtime"
UV_CACHE="$(absolute_path "${CDLAM_MODEL_UV_CACHE:-$ROOT/.cache/uv-model}")"
HTTP_TIMEOUT="${CDLAM_MODEL_HTTP_TIMEOUT:-300}"
[[ "$HTTP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail \
  "CDLAM_MODEL_HTTP_TIMEOUT must be a positive integer number of seconds"

case "$MODEL_ENV" in
  /|/bin|/etc|/home|/root|/tmp|/usr|/var|"$ROOT"|"$DEPS_ROOT")
    fail "refusing unsafe environment target: $MODEL_ENV"
    ;;
esac

UV_VERSION="$($PYTHON - "$LOCK" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["installer"]["uv_version"])
PY
)"
UV_EXTRA="$($PYTHON - "$LOCK" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["installer"]["upstream_extra"])
PY
)"
HEADLESS_OPENCV_VERSION="$($PYTHON - "$LOCK" <<'PY'
import json
import sys

print(
    json.load(open(sys.argv[1], encoding="utf-8"))["installer"]
    ["post_sync_reinstall"]["opencv-python-headless"]
)
PY
)"

if [[ "$DRY_RUN" == yes ]]; then
  if [[ ! -d "$ACWM_ROOT" ]]; then
    printf 'DRY RUN: CDLAM_ACCEPT_BASE_LICENSE=yes CDLAM_DEPS_DIR=%q PYTHON=%q bash %q base\n' \
      "$DEPS_ROOT" "$PYTHON" "$ROOT/scripts/fetch_optional_deps.sh"
  else
    printf 'DRY RUN: verify staged source %q\n' "$ACWM_ROOT"
  fi
  printf 'DRY RUN: create unified GPU environment %q with CPython 3.10\n' "$MODEL_ENV"
  printf 'DRY RUN: uv %s sync --locked --no-dev --extra %q against %q\n' \
    "$UV_VERSION" "$UV_EXTRA" "$ACWM_ROOT/uv.lock"
  printf 'DRY RUN: install locked runtime supplements from %q\n' "$EXTRA_LOCK"
  printf 'DRY RUN: reinstall opencv-python-headless==%s after overlapping OpenCV wheels\n' \
    "$HEADLESS_OPENCV_VERSION"
  printf 'DRY RUN: python %q --environment %q --acwm-root %q\n' \
    "$ROOT/scripts/model_runtime_doctor.py" "$MODEL_ENV" "$ACWM_ROOT"
  exit 0
fi

"$PYTHON" -m venv --help >/dev/null 2>&1 || fail "the Python venv module is unavailable"
command -v git >/dev/null 2>&1 || fail "git is required"
command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg is required"

if [[ ! -d "$ACWM_ROOT" ]]; then
  [[ "${CDLAM_ACCEPT_BASE_LICENSE:-}" == yes ]] || fail \
    "staged source is absent; review its license, then set CDLAM_ACCEPT_BASE_LICENSE=yes"
  CDLAM_DEPS_DIR="$DEPS_ROOT" PYTHON="$PYTHON" \
    bash "$ROOT/scripts/fetch_optional_deps.sh" base
fi

"$PYTHON" "$ROOT/scripts/model_runtime_doctor.py" \
  --source-only --acwm-root "$ACWM_ROOT"

if [[ -e "$MODEL_ENV" ]]; then
  [[ -d "$MODEL_ENV" ]] || fail "environment target is not a directory: $MODEL_ENV"
  "$PYTHON" "$ROOT/scripts/model_runtime_doctor.py" \
    --environment "$MODEL_ENV" --acwm-root "$ACWM_ROOT"
  echo "bootstrap_model_runtime: existing GPU environment verified; no files changed"
  exit 0
fi

BOOTSTRAP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cdlam-model-uv.XXXXXX")"
ENVIRONMENT_CREATED=no
cleanup() {
  status=$?
  rm -rf "$BOOTSTRAP_DIR"
  if [[ $status -ne 0 && "$ENVIRONMENT_CREATED" == yes && -d "$MODEL_ENV" ]]; then
    rm -rf "$MODEL_ENV"
  fi
  exit "$status"
}
trap cleanup EXIT

"$PYTHON" -m venv "$BOOTSTRAP_DIR"
"$BOOTSTRAP_DIR/bin/python" -m pip install \
  --disable-pip-version-check "uv==$UV_VERSION"
UV_BIN="$BOOTSTRAP_DIR/bin/uv"

mkdir -p "$(dirname "$MODEL_ENV")" "$UV_CACHE"
ENVIRONMENT_CREATED=yes
UV_PROJECT_ENVIRONMENT="$MODEL_ENV" \
UV_CACHE_DIR="$UV_CACHE" \
UV_HTTP_TIMEOUT="$HTTP_TIMEOUT" \
UV_LINK_MODE=copy \
  "$UV_BIN" sync \
    --locked \
    --no-dev \
    --extra "$UV_EXTRA" \
    --python "$PYTHON" \
    --no-python-downloads \
    --project "$ACWM_ROOT"

PYTORCH3D_NO_EXTENSION=1 \
UV_CACHE_DIR="$UV_CACHE" \
UV_HTTP_TIMEOUT="$HTTP_TIMEOUT" \
UV_LINK_MODE=copy \
  "$UV_BIN" pip install \
    --python "$MODEL_ENV/bin/python" \
    --no-deps \
    --no-build-isolation \
    --requirements "$EXTRA_LOCK"

# Both OpenCV distributions are pinned by the upstream graph and share the cv2
# payload path. Reinstalling the exact headless wheel last avoids an undeclared
# host libGL dependency while retaining both required metadata records.
UV_CACHE_DIR="$UV_CACHE" \
UV_HTTP_TIMEOUT="$HTTP_TIMEOUT" \
UV_LINK_MODE=copy \
  "$UV_BIN" pip install \
    --python "$MODEL_ENV/bin/python" \
    --no-deps \
    --reinstall \
    "opencv-python-headless==$HEADLESS_OPENCV_VERSION"

# The doctor checks every installed Requires-Dist entry and permits only the
# exact upstream metadata conflicts recorded in model_runtime.lock.json.
"$PYTHON" "$ROOT/scripts/model_runtime_doctor.py" \
  --environment "$MODEL_ENV" --acwm-root "$ACWM_ROOT"
ENVIRONMENT_CREATED=no

cat <<EOF

CD-LAM GPU environment base is ready.
  Python:   $MODEL_ENV/bin/python
  torchrun: $MODEL_ENV/bin/torchrun
  source:   $ACWM_ROOT

No model, tokenizer, text-encoder, dataset, or metric asset was downloaded.
setup.sh installs the remaining CD-LAM tools into this same
environment. After adding your licensed local assets, run:
  bash run.sh runtime-doctor --stage all
EOF
