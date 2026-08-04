# Phase 01 — Summary

## What was built

Reproducible CPU/CUDA/TPU development environment for the medfm framework:

- **Package scaffold** per `idea.md`: `medfm/{cli,core,data,models,peft,tasks,training,evaluation,inference,registry,tools}`
  with docstring-only placeholder `__init__.py` files (no empty abstractions), plus
  `configs/`, `scripts/`, `examples/`, `docker/`, `artifacts/` (git-ignored with `.gitkeep`).
- **Packaging**: `pyproject.toml` (hatchling) with pinned direct dependencies and extras:
  `medical`, `pathology`, `cucim`, `hf`, `cuda` (bitsandbytes), `tpu` (torch_xla[tpu]),
  `tracking` (tensorboard), `mlflow`, `wandb`, plus a `dev` dependency group.
  Matched accelerator set: **torch 2.9.0 / torchvision 0.24.0 / torch_xla[tpu] 2.9.0 /
  libtpu 0.0.21** (torch 2.9.0 ships CUDA 12.8 wheels). `uv.lock` committed (207 packages,
  Linux-only resolution). Lockfile update process documented in pyproject + README.
- **Developer commands**: `Makefile` with install / install-dev / install-tpu / lock /
  lint / typecheck / test / test-gpu / test-tpu / test-distributed-gpu /
  test-distributed-tpu / smoke / doctor; all fail nonzero. Hardware tests are protected
  by env-var guards (`MEDFM_RUN_GPU_TESTS`, `MEDFM_RUN_TPU_TESTS`,
  `MEDFM_RUN_DISTRIBUTED_TESTS`); enabling the guard without the hardware fails the test.
- **Diagnostics**: `medfm doctor` (`medfm/tools/doctor.py`) with `--backend
  auto|cpu|cuda|xla_tpu` and `--json`; reports Python/torch/CUDA/driver/GPU/VRAM,
  BF16, SDPA + FlashAttention, NCCL, monai/transformers/peft/bitsandbytes versions,
  disk/cache status; TPU mode reports torch_xla/libtpu, PJRT, device count/topology,
  XLA BF16, SPMD, and incompatible CUDA-only packages. JSON output conforms to
  `medfm/tools/doctor_schema.json`. No tokens/credentials; home paths masked.
- **Reproducibility**: `medfm/training/run_metadata.py` captures commit SHA, dirty
  state, lockfile hash, runtime + accelerator topology, seed, precision, batch
  geometry (validates `global = micro × world × accum`), param counts, peak memory,
  shape buckets, compiler flags, XLA metrics; dataset/preprocessing/base-model hashes
  are explicit placeholders. Canonical deterministic JSON + `config_hash()`.
- **Tracking**: `medfm/training/tracking.py` — `Tracker` protocol, `LocalJSONTracker`
  (default), `TensorBoardTracker` (optional extra), reserved MLflow/W&B adapters behind
  extras, recursive redaction of sensitive keys.
- **Containers**: `docker/Dockerfile` (CUDA dev, non-root, immutable-tag base),
  `docker/Dockerfile.ci` (CPU quality gate), `docker/compose.yaml` (explicit GPU
  reservation, named cache volumes), `docker/README.md` (driver compatibility, disk
  budget, GCP TPU VM provisioning: image, service account, storage, network — no
  credentials committed), `scripts/tpu_vm_bootstrap.sh`.
- **Validation**: `validate_phase` extended with `PHASE_01_REQUIRED_FILES`.

## Key decisions

- **uv** as package manager with a committed `uv.lock`; Linux-only resolution
  (`tool.uv.environments`) because CUDA/TPU wheels are Linux-only.
- **mypy** (not pyright), strict for first-party code; scoped relaxations only for
  tool modules calling untyped third-party APIs and for MONAI lazy exports.
- **ruff** line-length 120 (matches Phase 00 sources).
- Project license: **Apache-2.0** (anticipated by `docs/licensing_policy.md`).
- torch pinned to 2.9.0 exactly to match the newest torch_xla on PyPI (2.9.0);
  both must move together.
- MONAI 1.6 center-crop updates the affine origin (offset × spacing) and records the
  pre-crop affine under `original_affine` — tests and smoke assert both, which also
  satisfies the "never discard affine" rule.

## Test results

- CPU suite: 92 passed, 6 skipped (protected hardware guards with recorded reasons).
- Protected GPU suite (`make test-gpu`): BF16 allocation + hardware checks passed on
  RTX 4060 Laptop (sm 8.9); multi-device reduction skipped (single GPU, reason recorded).
- `make smoke`: 3/3 checks passed (doctor schema, MONAI 3D load/crop, tiny LoRA step).
- Fresh editable install in a clean venv (`/tmp/medfm-fresh-venv`): import + CLI JSON OK.
- TPU lock resolution audited: no bitsandbytes/cuCIM/FlashAttention in the `tpu` extra.

See `test_results.json` for the machine-readable record.
