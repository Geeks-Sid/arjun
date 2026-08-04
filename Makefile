# medfm developer commands. Every target exits nonzero on failure.
UV ?= uv
PYTEST := $(UV) run --frozen pytest

.PHONY: install install-dev install-tpu lint typecheck test test-gpu test-tpu \
        test-distributed-gpu test-distributed-tpu smoke doctor lock

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
