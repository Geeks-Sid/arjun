#!/usr/bin/env bash
# Bootstrap a medfm development environment on a Google Cloud TPU VM.
#
# Prereqs (see docker/README.md "TPU VM provisioning"):
#   - TPU VM created with a tpu-ubuntu2204-base (or newer) image
#   - repo cloned at $REPO_DIR
#   - service account with storage access attached to the TPU VM
#
# This script never installs bitsandbytes, FlashAttention, or cuCIM:
# the TPU baseline is torch==2.9.0 + torch_xla[tpu]==2.9.0 (libtpu).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/arjun}"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.13
uv sync --frozen --no-dev --extra medical --extra hf --extra tpu

export PJRT_DEVICE=TPU

echo "== TPU doctor =="
uv run --frozen python -m medfm.tools.doctor --backend xla_tpu

echo "== TPU smoke (one-step BF16 optimization on all local devices) =="
MEDFM_RUN_TPU_TESTS=1 uv run --frozen pytest tests/ -q -m tpu
