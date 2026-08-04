# Phase 01 → Phase 02 Handoff

## Supported versions (recorded)

- Python **3.11–3.13** (3.13 reference; host runs 3.13.13).
- **torch 2.9.0 / torchvision 0.24.0** (CUDA 12.8 wheels), matched with
  **torch_xla[tpu] 2.9.0 + libtpu 0.0.21** — upgrade only as a set.
- Host NVIDIA driver 595.84; minimum documented 525.60.13, recommended >= 545.
- Linux x86_64 only (`tool.uv.environments` in pyproject.toml).

## Optional dependencies unavailable on the target host

- `flash-attn` — not installed (SDPA with flash SDP kernel is available and is the default path).
- `cucim-cu12`, `mlflow`, `wandb`, `openslide`/`tiffslide` runtime use — extras exist but
  are not part of install-dev workflows exercised here (pathology extra is locked, not smoke-tested).
- `torch_xla`/`libtpu` — TPU-only; not installed on the CUDA dev host by design.

## Tested runtime matrices and launch commands

- CPU: `make test` (92 passed, 6 protected skips) — torch 2.9.0+cu128, no CUDA init.
- CUDA single-device: `make test-gpu` on RTX 4060 Laptop 8 GB (sm 8.9, BF16 OK, NCCL OK).
- TPU: not run here; launch with `scripts/tpu_vm_bootstrap.sh` then `make test-tpu`
  (PJRT_DEVICE=TPU) on a `tpu-ubuntu2204-base` VM.
- Distributed: `make test-distributed-gpu` / `make test-distributed-tpu` on multi-device runners.

## Package conventions for Phase 02

- First-party code lives under `medfm/<area>/`; Phase 02 types go in `medfm/core/`.
- mypy strict is enforced repo-wide (`make typecheck`); keep new modules strict-clean —
  no new global overrides. Third-party untyped imports go in the existing
  `ignore_missing_imports` override list.
- ruff line-length 120, rules E/F/I/UP/B/W; format checked in `make lint`.
- No top-level imports of `bitsandbytes`, `torch_xla`, `flash_attn`, or `cucim`
  (enforced by `tests/phase_01/test_packaging.py`); lazy-import inside
  backend/capability modules.
- Tests live in `tests/phase_<NN>/`; hardware tests use the `gpu` / `tpu` /
  `distributed` markers (guards in `tests/phase_01/conftest.py`).
- Register each phase's required files in `medfm/tools/validate_phase.py`
  (`PHASE_<NN>_REQUIRED_FILES`) and smoke checks in `medfm/tools/smoke.py::SMOKE_CHECKS`.

## Run-metadata API contract

`medfm.training.run_metadata.capture_run_metadata(*, accelerator_backend, seed,
precision, microbatch_per_device, gradient_accumulation_steps=1, world_size=None,
model=None, base_model_revision=None, adapter_config=None,
dataset_manifest_sha256=None, preprocessing_config=None, shape_buckets=None,
compiler_flags=None, extra=None) -> RunMetadata`

- `accelerator_backend`: `"cpu" | "cuda" | "xla_tpu"`.
- Returns a frozen `RunMetadata` dataclass; `to_canonical_json()` is deterministic
  (sorted keys, fixed separators); `config_hash()` hashes configuration only
  (excludes `peak_memory_bytes` and `xla_metrics_report`).
- `effective_batch_size = microbatch_per_device * world_size * gradient_accumulation_steps`
  (validated positive; world_size defaults to local device count).
- Later phases fill `dataset_manifest_sha256` (Phase 03), `preprocessing_config`
  (Phase 04), and `base_model_revision` / `adapter_config` (Phases 05/10).

## Tracker contract

`medfm.training.tracking.Tracker` protocol: `log_params(dict)`,
`log_metrics(dict[str, float], step: int)`, `close()`. Default
`LocalJSONTracker(log_dir, sensitive_fragments=...)`; `create_tracker("local_json" |
"tensorboard" | "mlflow" | "wandb", **kwargs)`. All trackers redact keys matching
sensitive fragments (token/secret/password/api_key/credential/patient/mrn).

## Environment variables

- `MEDFM_MODEL_CACHE`, `MEDFM_DATASET_CACHE` — cache locations (default `~/.cache/medfm/...`).
- `MEDFM_RUN_GPU_TESTS=1`, `MEDFM_RUN_TPU_TESTS=1`, `MEDFM_RUN_DISTRIBUTED_TESTS=1` —
  protected-test guards (set by the corresponding `make` targets).
- `PJRT_DEVICE=TPU` — XLA runtime selection for TPU runs.
