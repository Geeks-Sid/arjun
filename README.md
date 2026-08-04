# medfm — unified medical foundation-model framework

One framework for training and evaluating medical foundation models across
2D radiology, 3D CT/MRI, whole-slide pathology, and vision-language tasks —
with LoRA/QLoRA-first PEFT training on CPU (tests), NVIDIA CUDA GPUs, and
Google Cloud TPUs (PyTorch/XLA).

Status: Phase 01 (repository, environment, reproducibility) complete.
See `implementation_plan/` for the phase-gated plan and `docs/` for the
frozen v1 scope.

## Supported environment matrix

| Component | Pinned version |
|---|---|
| Python | 3.11–3.13 (3.13 reference) |
| PyTorch / torchvision | 2.9.0 / 0.24.0 (CUDA 12.8 wheels) |
| PyTorch/XLA (TPU extra) | 2.9.0 with libtpu 0.0.21 |
| Host NVIDIA driver | >= 545 recommended (>= 525.60.13 minimum) |
| Platform | Linux x86_64 only |

The torch and torch_xla versions are pinned as a matched set — do not
upgrade one without the other.

## Setup

```bash
make install        # runtime: core + medical + HF/PEFT extras
make install-dev    # everything local: + pathology, cuda (bitsandbytes), tracking, dev tools
make install-tpu    # TPU VM baseline (torch_xla[tpu]; NO bitsandbytes/cuCIM/FlashAttention)
```

Dependency policy: direct dependencies are pinned or lower-bounded in
`pyproject.toml`; exact resolution lives in the committed `uv.lock`. To
update: `uv lock --upgrade-package <pkg>`, re-run the acceptance command,
commit both files. cuCIM, FlashAttention, bitsandbytes, MLflow, and W&B are
optional extras and never part of the CPU or TPU baseline.

## Developer commands

```bash
make lint                  # ruff check + format check
make typecheck             # mypy (strict for first-party code)
make test                  # CPU tests; protected hardware markers excluded
make test-gpu              # CUDA tests (requires local GPU; env-guarded)
make test-tpu              # TPU tests (PJRT_DEVICE=TPU; env-guarded)
make test-distributed-gpu  # multi-GPU protected job
make test-distributed-tpu  # multi-device TPU protected job
make smoke                 # phase smoke checks
make doctor                # runtime diagnostics (also: medfm doctor --json)
```

Every command exits nonzero on failure.

## Layout

```
medfm/          framework package (cli, core, data, models, peft, tasks,
                training, evaluation, inference, registry, tools)
tests/          phase-scoped test suites (tests/phase_NN/)
docker/         CUDA dev image, CI image, compose file
scripts/        environment bootstrap scripts (e.g. TPU VM)
model_registry/ license registry and v1 model scope (Phase 00)
agent/          phase-gated execution protocol and reports
docs/           product, governance, and architecture (ADR) documents
configs/        run configurations (populated in later phases)
artifacts/      run outputs — git-ignored
```

## Reproducibility and tracking

`medfm.training.run_metadata.capture_run_metadata()` records commit SHA,
dirty-tree state, lockfile hash, runtime/accelerator details, seed, precision,
batch geometry, and configuration hashes as canonical (deterministic) JSON.
`medfm.training.tracking` provides a local-first `Tracker` protocol
(`LocalJSONTracker` default; TensorBoard optional; MLflow/W&B reserved behind
extras) with automatic redaction of sensitive keys.

## License

Apache-2.0 (see `LICENSE`). Model *weights* are governed per-record in
`model_registry/licenses.yaml` — see `docs/licensing_policy.md`.
