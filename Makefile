PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RUFF ?= $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
PYTHONPATH_ROOT := $(CURDIR)/src
TRAIN_SMOKE_ROOT ?= build/train-smoke

.DEFAULT_GOAL := help

.PHONY: help check release-check lint test smoke data-smoke train-smoke clean-wheel wheel

help:
	@echo "CD-LAM release targets:"
	@echo "  make check         Run all CPU source-release gates"
	@echo "  make release-check Check language, paths, syntax, and release metadata"
	@echo "  make lint          Run Ruff"
	@echo "  make test          Run unit tests"
	@echo "  make smoke         Run the deterministic core smoke test"
	@echo "  make data-smoke    Build and validate the portable test dataset"
	@echo "  make train-smoke   Run all four synthetic training stages on CPU"
	@echo "  make wheel         Build the wheel into dist/"

check: release-check lint test smoke data-smoke train-smoke

release-check:
	$(PYTHON) scripts/release_check.py --strict

lint:
	$(RUFF) check .

test:
	$(PYTHON) -m pytest -q

smoke:
	PYTHONPATH="$(PYTHONPATH_ROOT)" $(PYTHON) -m cd_lam smoke

data-smoke:
	rm -rf "$(TRAIN_SMOKE_ROOT)/data"
	PYTHONPATH="$(PYTHONPATH_ROOT)" $(PYTHON) -m cd_lam data-prepare \
		--input tests/fixtures/episodes.jsonl \
		--output "$(TRAIN_SMOKE_ROOT)/data"
	PYTHONPATH="$(PYTHONPATH_ROOT)" $(PYTHON) -m cd_lam data-validate \
		--root "$(TRAIN_SMOKE_ROOT)/data"

train-smoke:
	PYTHONPATH="$(PYTHONPATH_ROOT)" $(PYTHON) -m cd_lam train-smoke \
		--output-root "$(TRAIN_SMOKE_ROOT)" \
		--steps 2 \
		--json

clean-wheel:
	rm -rf build dist

wheel: clean-wheel
	$(PYTHON) -m build --wheel --no-isolation --outdir dist
	$(PYTHON) scripts/check_wheel.py dist/*.whl
