# medfm developer commands. Every target exits nonzero on failure.
UV ?= uv
PYTEST := $(UV) run --frozen pytest

.PHONY: install install-dev install-tpu lint typecheck test test-gpu test-tpu \
        test-distributed-gpu test-distributed-tpu smoke doctor lock \
        test-level1 test-level2 test-golden test-protected coverage build \
        security release-check release-matrix ci

# Runtime + medical + HF/PEFT extras (CPU/CUDA host).
install:
	$(UV) sync --frozen --no-dev --extra medical --extra hf

# Full development environment (all local extras + dev tools).
install-dev:
	$(UV) sync --frozen --extra medical --extra pathology --extra hf \
		--extra cuda --extra tracking --all-groups

# TPU VM baseline: NO bitsandbytes / cuCIM / FlashAttention.
install-tpu:
	$(UV) sync --frozen --no-dev --extra medical --extra hf --extra tpu

lock:
	$(UV) lock

lint:
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .

typecheck:
	$(UV) run --frozen mypy

# Tier 0: CPU contract tests. Protected-hardware markers are deselected by
# their conftest guards; this target must pass on a machine with no GPU.
test:
	$(PYTEST) tests/ -q

# Phase 18 release support matrix (checklist items under "Test matrix").
# Level 1: CPU contracts (schemas, registries, configs, losses, metrics,
# prompts, cache keys, coordinates).
test-level1:
	$(PYTEST) tests/ -q -m "level1 or golden or security or reliability"

# Level 2: synthetic-GPU / tiny-model accelerator smoke (routine accelerator
# jobs use tiny models only). Guarded by MEDFM_RUN_GPU_TESTS.
test-level2:
	MEDFM_RUN_GPU_TESTS=1 $(PYTEST) tests/ -q -m level2

# Level 4: golden regression (CPU; pinned fixtures, dtype-aware tolerances).
test-golden:
	$(PYTEST) tests/phase_18/test_golden_regression.py -q

# Level 3: protected real-checkpoint smoke. Guarded by
# MEDFM_RUN_REAL_CHECKPOINTS; runs only against gated real-model checkpoints.
test-protected:
	MEDFM_RUN_REAL_CHECKPOINTS=1 $(PYTEST) tests/ -q -m "level3 or real_checkpoint"

# Tier 1/2/4: protected hardware jobs — opt-in via env var, fail if the
# hardware is absent rather than silently skipping.
test-gpu:
	MEDFM_RUN_GPU_TESTS=1 $(PYTEST) tests/ -q -m gpu

test-tpu:
	MEDFM_RUN_TPU_TESTS=1 PJRT_DEVICE=TPU $(PYTEST) tests/ -q -m tpu

test-distributed-gpu:
	MEDFM_RUN_DISTRIBUTED_TESTS=1 MEDFM_RUN_GPU_TESTS=1 $(PYTEST) tests/ -q -m "distributed and gpu"

test-distributed-tpu:
	MEDFM_RUN_DISTRIBUTED_TESTS=1 MEDFM_RUN_TPU_TESTS=1 PJRT_DEVICE=TPU \
		$(PYTEST) tests/ -q -m "distributed and tpu"

smoke:
	$(UV) run --frozen python -m medfm.tools.smoke --phase 01

doctor:
	$(UV) run --frozen python -m medfm.tools.doctor

# Coverage: machine-readable xml+coverage artifacts for CI (requires pytest-cov).
coverage:
	$(PYTEST) tests/ -q --cov=medfm --cov-report=xml:artifacts/coverage/coverage.xml \
		--cov-report=term-missing

# Package build (release artifact).
build:
	$(UV) build --out-dir artifacts/dist

# Static checks + release gate (Phase 18).
security:
	$(UV) run --frozen python scripts/scan_secrets.py --root .
	$(UV) run --frozen ruff check .

release-check:
	$(UV) run --frozen python -m medfm.cli.release validate

release-matrix:
	$(UV) run --frozen python -m medfm.cli.release matrix

# Phase 18 acceptance gate (see implementation_plan/phase_18_ci_hardening_and_release.md).
ci: lint typecheck test security release-check
