#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3.10}"
VENV="${CDLAM_VENV:-$ROOT/.venv}"
DEPS_ROOT="${CDLAM_DEPS_DIR:-$ROOT/.deps}"
GPU="${CDLAM_GPU:-0}"
WITH_METRICS=0
WITH_MODELS=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: CDLAM_ACCEPT_BASE_LICENSE=yes bash scripts/bootstrap.sh [OPTIONS]

Create CD-LAM's single pinned GPU environment, stage the verified upstream
runtime, install the data/download/test tools, validate CUDA, and run the
release smoke suite. The environment contract is CPython 3.10, PyTorch
2.7.0+cu128, and CUDA 12.8 on Linux x86-64.

Options:
  --gpu INDEX       CUDA device used by the driver and optimizer smoke (default: 0).
  --with-models     Download and verify all three released model entries.
  --with-metrics    Fetch the optional SAM3 and CoWTracker source repositories.
  --dry-run         Print setup operations without creating files.
  -h, --help        Show this help.

Model weights, datasets, tokenizer/text-encoder assets, SAM3 weights, and
CoWTracker weights are not downloaded unless their explicit path is requested.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || { echo "--gpu requires an index" >&2; exit 2; }
      GPU="$2"
      shift
      ;;
    --with-metrics) WITH_METRICS=1 ;;
    --with-models) WITH_MODELS=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be a non-negative integer" >&2; exit 2; }
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "CPython 3.10 executable not found: $PYTHON" >&2
  exit 2
}

runtime_args=(
  --python "$PYTHON"
  --environment "$VENV"
  --deps-root "$DEPS_ROOT"
)
if [[ "$DRY_RUN" == 1 ]]; then
  runtime_args+=(--dry-run)
fi
bash "$ROOT/scripts/bootstrap_model_runtime.sh" "${runtime_args[@]}"

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY RUN: install CD-LAM test, data, and download tools into %q\n' "$VENV"
  printf 'DRY RUN: validate torch 2.7.0+cu128, CUDA 12.8, and GPU %q\n' "$GPU"
  printf 'DRY RUN: run release, test, data, integration, and CUDA optimizer smokes\n'
  if [[ "$WITH_MODELS" == 1 ]]; then
    "$PYTHON" "$ROOT/scripts/download_models.py" --dry-run
  fi
  if [[ "$WITH_METRICS" == 1 ]]; then
    printf 'DRY RUN: fetch optional SAM3 and CoWTracker sources\n'
  fi
  exit 0
fi

UV_VERSION="$($PYTHON - "$ROOT/configs/model_runtime.lock.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["installer"]["uv_version"])
PY
)"
BOOTSTRAP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cdlam-release-uv.XXXXXX")"
cleanup() {
  status=$?
  rm -rf "$BOOTSTRAP_DIR"
  exit "$status"
}
trap cleanup EXIT

"$PYTHON" -m venv "$BOOTSTRAP_DIR"
"$BOOTSTRAP_DIR/bin/python" -m pip install \
  --disable-pip-version-check "uv==$UV_VERSION"
UV_BIN="$BOOTSTRAP_DIR/bin/uv"
UV_CACHE="${CDLAM_MODEL_UV_CACHE:-$ROOT/.cache/uv-model}"
UV_HTTP_TIMEOUT="${CDLAM_MODEL_HTTP_TIMEOUT:-300}"
mkdir -p "$UV_CACHE"
UV_CACHE_DIR="$UV_CACHE" UV_HTTP_TIMEOUT="$UV_HTTP_TIMEOUT" UV_LINK_MODE=copy \
  "$UV_BIN" pip install \
  --python "$VENV/bin/python" \
  --requirements "$ROOT/requirements.lock"
UV_CACHE_DIR="$UV_CACHE" UV_HTTP_TIMEOUT="$UV_HTTP_TIMEOUT" UV_LINK_MODE=copy \
  "$UV_BIN" pip install \
  --python "$VENV/bin/python" \
  --no-deps \
  --no-build-isolation \
  "$ROOT[test,data,download]"

"$PYTHON" "$ROOT/scripts/model_runtime_doctor.py" \
  --environment "$VENV" \
  --acwm-root "$DEPS_ROOT/acwm-runtime" \
  --check-driver \
  --gpu "$GPU"
"$VENV/bin/python" "$ROOT/scripts/gpu_smoke.py" --gpu "$GPU"

if [[ "$WITH_METRICS" == 1 ]]; then
  bash "$ROOT/scripts/fetch_optional_deps.sh" metrics
fi

"$VENV/bin/python" "$ROOT/scripts/release_check.py" --strict
CDLAM_TEST_ACWM_ROOT="$DEPS_ROOT/acwm-runtime" \
  "$VENV/bin/python" -m pytest -q "$ROOT/tests"
"$VENV/bin/python" -m cd_lam smoke
"$VENV/bin/python" -m cd_lam data-prepare \
  --input "$ROOT/tests/fixtures/episodes.jsonl" \
  --output "$VENV/test-data"
"$VENV/bin/python" -m cd_lam data-validate --root "$VENV/test-data"
CUDA_VISIBLE_DEVICES="$GPU" "$VENV/bin/python" -m cd_lam train-smoke \
  --output-root "$VENV/train-smoke" --steps 1

if [[ "$WITH_MODELS" == 1 ]]; then
  "$VENV/bin/python" "$ROOT/scripts/download_models.py" \
    --local-dir "$ROOT/artifacts"
fi

cat <<EOF

CD-LAM is ready.
  environment: $VENV
  Python:      $VENV/bin/python
  torchrun:    $VENV/bin/torchrun
  CUDA:        12.8
  PyTorch:     2.7.0+cu128
  GPU:         $GPU
  source:      $DEPS_ROOT/acwm-runtime

Next:
  cp configs/runtime.example.json configs/runtime.json
  bash run.sh runtime-doctor --stage all
EOF
