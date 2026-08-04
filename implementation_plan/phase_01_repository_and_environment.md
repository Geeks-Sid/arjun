# Phase 01: Repository, Environment, and Reproducibility

## Objective

Create reproducible CPU, CUDA, and PyTorch/XLA environments with protected GPU and TPU smoke tests.

## Dependencies

- [x] Phase 00 is accepted.
- [x] Supported Python and CUDA versions are chosen and documented.
- [x] A matched PyTorch/PyTorch-XLA/libtpu version set is chosen from supported TPU runtime versions.
- [x] Package manager and lockfile strategy are approved.

## Scope boundaries

Allowed areas: repository scaffolding, packaging, containers, CI bootstrap, `medfm/tools/doctor.py`, and environment tests.

Do not integrate real foundation-model checkpoints or implement task behavior.

## Implementation checklist

### Repository scaffold

- [x] Create the package and directory layout described in `idea.md`.
- [x] Create `pyproject.toml` with core, medical, pathology, HF/PEFT, quality, tracking, and optional GPU groups.
- [x] Generate and commit a reproducible lockfile.
- [x] Add `README.md`, `LICENSE`, `.gitignore`, and `Makefile`.
- [x] Add placeholders only where package discovery requires them; avoid empty architecture abstractions.
- [x] Add `artifacts/`, model cache, dataset cache, and patient-data patterns to `.gitignore`.

### Dependency policy

- [x] Pin direct dependencies and define a process for lockfile updates.
- [x] Make cuCIM and hardware-specific acceleration optional.
- [x] Ensure CPU tests do not import bitsandbytes or CUDA-only libraries eagerly.
- [x] Keep external experiment trackers optional and disabled by default.
- [x] Select mypy or pyright and define strictness expectations by package.
- [x] Define separate dependency extras/lock resolution for `cuda` and `tpu`; do not install bitsandbytes in the TPU baseline.
- [x] Pin compatible `torch` and `torch_xla[tpu]` versions together.
- [x] Keep custom CUDA packages such as FlashAttention and cuCIM out of the TPU dependency set.

### Developer commands

- [x] Implement `make install` and `make install-dev`.
- [x] Implement `make lint`, `make typecheck`, and `make test`.
- [x] Implement `make test-gpu` with a clear hardware marker.
- [x] Implement `make test-tpu` with a clear hardware marker and PJRT launch configuration.
- [x] Implement `make test-distributed-gpu` and `make test-distributed-tpu` protected jobs.
- [x] Implement `make smoke` and `make doctor`.
- [x] Ensure every command returns a nonzero status on failure.

### Runtime diagnostics

- [x] Implement `medfm doctor` output for Python, PyTorch, CUDA, driver, GPU, and VRAM.
- [x] Report BF16 support and SDPA/FlashAttention availability.
- [x] Report bitsandbytes, MONAI, Transformers, and PEFT versions.
- [x] Report free disk, model cache, dataset cache, and writable status.
- [x] Avoid printing credentials, tokens, or patient-data paths.
- [x] Provide machine-readable JSON output in addition to human-readable output.
- [x] Add backend selection and report CUDA device count/NCCL availability or TPU device count/PJRT runtime.
- [x] Report PyTorch-XLA/libtpu versions, TPU type/topology, XLA BF16 support, and SPMD availability.
- [x] Report incompatible packages and custom CUDA extensions when TPU is selected.

### Containers and local parity

- [x] Add development and CI Dockerfiles.
- [x] Add a compose file with explicit GPU configuration and mounted cache locations.
- [x] Run containers as a non-root user where practical.
- [x] Pin the base image by digest or immutable version.
- [x] Document host-driver compatibility and expected disk requirements.
- [x] Add a CUDA image and a TPU VM/bootstrap environment; do not assume the CUDA image runs on TPU.
- [x] Document Google Cloud TPU VM provisioning, storage, service-account, and network prerequisites without committing credentials.

### Reproducibility capture

- [x] Implement a utility that captures commit SHA and dirty status.
- [x] Capture lockfile hash, CUDA/driver versions, GPU model, seed, and precision.
- [x] Define placeholders for future dataset, preprocessing, and base-model hashes.
- [x] Capture trainable parameter counts, effective batch size, and peak VRAM.
- [x] Serialize run metadata deterministically.
- [x] Capture accelerator backend, topology, world size, compiler/runtime flags, precision policy, and shape-bucket set.
- [x] Capture XLA compilation/metrics reports for TPU runs and NCCL/CUDA runtime details for distributed GPU runs.

### Local tracking

- [x] Define a small tracker protocol.
- [x] Implement a local JSON tracker first.
- [x] Add TensorBoard support without making it mandatory.
- [x] Reserve optional MLflow and W&B adapters behind extras.
- [x] Redact configured sensitive keys before logging.

## Tests and verification

- [x] Test a clean editable install in a fresh environment.
- [x] Run imports on a CPU-only process.
- [x] Run one synthetic MONAI 3D load/crop pipeline.
- [x] Run one minimal LoRA optimization step using a tiny local model.
- [x] On the target GPU, allocate BF16 tensors and report peak memory.
- [x] Verify ignored data/model files are not considered for commit.
- [x] Verify doctor JSON conforms to a schema.
- [ ] Run a one-step BF16 linear-model optimization on every local TPU device. *(implemented in `tests/phase_01/test_tpu.py`, protected; no TPU hardware on this workstation — run via `make test-tpu` on a TPU VM)*
- [ ] Run a multi-device reduction test on CUDA and TPU protected runners. *(implemented and protected; single-GPU host, no TPU — run on multi-device runners)*
- [x] Verify CPU-only import does not initialize CUDA or XLA.
- [x] Verify CUDA-only extensions are not imported by the TPU environment.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [PyTorch/XLA installation and quick start](https://docs.pytorch.org/xla/master/)
- [Cloud TPU PyTorch quick start](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)

## Smoke command

```bash
make smoke
```

## Acceptance command

```bash
make lint && make typecheck && make test && python -m medfm.tools.validate_phase --phase 01
```

## Exit criteria

- [x] CPU tests pass without downloading weights.
- [x] Protected GPU smoke confirms BF16 execution.
- [ ] Protected TPU smoke confirms XLA BF16 execution on all local devices. *(no TPU hardware on this workstation; test implemented and protected, acceptance recorded as `not_applicable` with justification in `agent/reports/phase_01/acceptance.json`)*
- [x] Synthetic NIfTI preprocessing and tiny LoRA step pass.
- [x] Runtime diagnostics are useful and redact sensitive values.
- [x] No data, weights, caches, or credentials are tracked.

## Handoff

- [x] Record exact supported Python/CUDA versions.
- [x] Record optional dependencies unavailable on the target host.
- [x] Record the tested CUDA and TPU runtime matrices and launch commands.
- [x] Document package conventions for Phase 02.
- [x] Provide the run-metadata API contract.
