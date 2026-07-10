#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${CDLAM_VENV:-$ROOT/.venv}"
WITH_METRICS=0
WITH_MODELS=0

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap.sh [--with-metrics] [--with-models]

Creates a local environment, installs CD-LAM, and runs the deterministic
smoke suite. Optional metric backends have separate licenses and are fetched
only when --with-metrics is requested. --with-models downloads the complete
public model snapshot (about 23 GB) after the source gates pass.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-metrics) WITH_METRICS=1 ;;
    --with-models) WITH_MODELS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "Python executable not found: $PYTHON" >&2
  exit 2
}

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv --system-site-packages "$VENV"
fi

if [[ "${CDLAM_UPGRADE_PIP:-no}" == yes ]]; then
  "$VENV/bin/python" -m pip install --upgrade pip
fi
install_target="$ROOT[test]"
if [[ "$WITH_MODELS" == 1 ]]; then
  install_target="$ROOT[test,download]"
fi
required_modules=(numpy torch yaml pytest ruff build)
if [[ "$WITH_MODELS" == 1 ]]; then
  required_modules+=(huggingface_hub)
fi
if "$VENV/bin/python" - "${required_modules[@]}" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print("missing Python modules: " + ", ".join(missing))
    raise SystemExit(1)
PY
then
  # Existing research runtimes often have a working CUDA torch build whose
  # wheel metadata would make pip resolve another multi-GB CUDA stack. Keep
  # that tested runtime intact and install only CD-LAM itself. Build isolation
  # still installs the small backend declared in pyproject.toml, so a newly
  # created venv is not coupled to its bundled setuptools version.
  "$VENV/bin/python" -m pip install --no-deps -e "$install_target"
else
  "$VENV/bin/python" -m pip install -e "$install_target"
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
  next:        bash scripts/run.sh doctor --strict
EOF
