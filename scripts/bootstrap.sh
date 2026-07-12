#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${CDLAM_VENV:-$ROOT/.venv}"
WITH_METRICS=0
WITH_MODELS=0
OFFLINE_CACHE=""
REUSE_SYSTEM_RUNTIME=0

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap.sh [OPTIONS]

Creates a local environment, installs CD-LAM, and runs the deterministic
smoke suite. Optional metric backends have separate licenses and are fetched
only when --with-metrics is requested. --with-models downloads the pinned
compact model snapshot after the source gates pass.

Options:
  --offline-cache PATH    Install without network access from a cache created
                          by scripts/offline_cache.py on a compatible machine.
  --reuse-system-runtime  Explicitly expose the parent interpreter's packages
                          inside the venv. This is not a clean-room install.
  --with-metrics          Fetch optional metric source after core validation.
  --with-models           Download and verify the public model snapshot.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-metrics) WITH_METRICS=1 ;;
    --with-models) WITH_MODELS=1 ;;
    --offline-cache)
      [[ $# -ge 2 ]] || { echo "--offline-cache requires a path" >&2; exit 2; }
      OFFLINE_CACHE="$2"
      shift
      ;;
    --reuse-system-runtime) REUSE_SYSTEM_RUNTIME=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "Python executable not found: $PYTHON" >&2
  exit 2
}

if [[ "$WITH_MODELS" == 1 ]]; then
  "$PYTHON" "$ROOT/scripts/download_models.py" --dry-run >/dev/null
fi

if [[ -n "$OFFLINE_CACHE" && ( "$WITH_METRICS" == 1 || "$WITH_MODELS" == 1 ) ]]; then
  echo "--offline-cache cannot fetch metrics or models; stage those licensed assets separately." >&2
  exit 2
fi
if [[ -n "$OFFLINE_CACHE" && "$REUSE_SYSTEM_RUNTIME" == 1 ]]; then
  echo "--offline-cache and --reuse-system-runtime are mutually exclusive." >&2
  exit 2
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  if [[ "$REUSE_SYSTEM_RUNTIME" == 1 ]]; then
    "$PYTHON" -m venv --system-site-packages "$VENV"
  else
    "$PYTHON" -m venv "$VENV"
  fi
fi

if [[ "${CDLAM_UPGRADE_PIP:-no}" == yes ]]; then
  "$VENV/bin/python" -m pip install --upgrade pip
fi
if [[ -n "$OFFLINE_CACHE" ]]; then
  include_system="$(sed -n 's/^include-system-site-packages = //p' "$VENV/pyvenv.cfg")"
  if [[ "$include_system" != false ]]; then
    echo "offline bootstrap requires an isolated venv; remove $VENV and retry." >&2
    exit 2
  fi
  "$PYTHON" "$ROOT/scripts/offline_cache.py" install \
    --cache "$OFFLINE_CACHE" --target-python "$VENV/bin/python"
else
  install_target="$ROOT[test]"
  if [[ "$WITH_MODELS" == 1 ]]; then
    install_target="$ROOT[test,download]"
  fi
  required_modules=(numpy torch yaml pytest ruff build setuptools wheel)
  if [[ "$WITH_MODELS" == 1 ]]; then
    required_modules+=(huggingface_hub)
  fi
  if [[ "$REUSE_SYSTEM_RUNTIME" == 1 ]] && "$VENV/bin/python" - "${required_modules[@]}" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print("missing Python modules: " + ", ".join(missing))
    raise SystemExit(1)
PY
  then
    # Reuse is explicit because resolving against an existing CUDA runtime is
    # useful for researchers but is not evidence of a clean installation.
    "$VENV/bin/python" -m pip install --no-build-isolation --no-deps -e "$install_target"
  else
    "$VENV/bin/python" -m pip install -r "$ROOT/requirements.lock"
    "$VENV/bin/python" -m pip install \
      --no-build-isolation --no-deps -e "$install_target"
  fi
fi

if [[ "$WITH_METRICS" == 1 ]]; then
  bash "$ROOT/scripts/fetch_optional_deps.sh" metrics
fi

"$VENV/bin/python" "$ROOT/scripts/release_check.py" --strict
"$VENV/bin/python" -m pytest -q "$ROOT/tests"
"$VENV/bin/python" -m cd_lam smoke
"$VENV/bin/python" -m cd_lam data-prepare \
  --input "$ROOT/tests/fixtures/episodes.jsonl" \
  --output "$VENV/test-data"
"$VENV/bin/python" -m cd_lam data-validate --root "$VENV/test-data"
"$VENV/bin/python" -m cd_lam train-smoke \
  --output-root "$VENV/train-smoke" --steps 1

if [[ "$WITH_MODELS" == 1 ]]; then
  "$VENV/bin/python" "$ROOT/scripts/download_models.py" \
    --local-dir "$ROOT/artifacts"
fi

cat <<EOF

CD-LAM is ready.
  environment: $VENV
  install:     $([[ -n "$OFFLINE_CACHE" ]] && echo offline-cache || ([[ "$REUSE_SYSTEM_RUNTIME" == 1 ]] && echo reused-runtime || echo isolated-online))
  next:        bash scripts/run.sh doctor --strict
EOF
