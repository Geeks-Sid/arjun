# Phase 01: Repository, Environment, and Reproducibility

## Objective

Create reproducible CPU, CUDA, and PyTorch/XLA environments with protected GPU and TPU smoke tests.

## Dependencies

- [ ] Phase 00 is accepted.
- [ ] Supported Python and CUDA versions are chosen and documented.
- [ ] A matched PyTorch/PyTorch-XLA/libtpu version set is chosen from supported TPU runtime versions.
- [ ] Package manager and lockfile strategy are approved.

## Scope boundaries

Allowed areas: repository scaffolding, packaging, containers, CI bootstrap, `medfm/tools/doctor.py`, and environment tests.

Do not integrate real foundation-model checkpoints or implement task behavior.

## Implementation checklist

### Repository scaffold

- [ ] Create the package and directory layout described in `idea.md`.
- [ ] Create `pyproject.toml` with core, medical, pathology, HF/PEFT, quality, tracking, and optional GPU groups.
- [ ] Generate and commit a reproducible lockfile.
- [ ] Add `README.md`, `LICENSE`, `.gitignore`, and `Makefile`.
- [ ] Add placeholders only where package discovery requires them; avoid empty architecture abstractions.
- [ ] Add `artifacts/`, model cache, dataset cache, and patient-data patterns to `.gitignore`.

### Dependency policy

- [ ] Pin direct dependencies and define a process for lockfile updates.
- [ ] Make cuCIM and hardware-specific acceleration optional.
- [ ] Ensure CPU tests do not import bitsandbytes or CUDA-only libraries eagerly.
- [ ] Keep external experiment trackers optional and disabled by default.
- [ ] Select mypy or pyright and define strictness expectations by package.
- [ ] Define separate dependency extras/lock resolution for `cuda` and `tpu`; do not install bitsandbytes in the TPU baseline.
- [ ] Pin compatible `torch` and `torch_xla[tpu]` versions together.
- [ ] Keep custom CUDA packages such as FlashAttention and cuCIM out of the TPU dependency set.

### Developer commands

- [ ] Implement `make install` and `make install-dev`.
- [ ] Implement `make lint`, `make typecheck`, and `make test`.
- [ ] Implement `make test-gpu` with a clear hardware marker.
- [ ] Implement `make test-tpu` with a clear hardware marker and PJRT launch configuration.
- [ ] Implement `make test-distributed-gpu` and `make test-distributed-tpu` protected jobs.
- [ ] Implement `make smoke` and `make doctor`.
- [ ] Ensure every command returns a nonzero status on failure.

### Runtime diagnostics

- [ ] Implement `medfm doctor` output for Python, PyTorch, CUDA, driver, GPU, and VRAM.
- [ ] Report BF16 support and SDPA/FlashAttention availability.
- [ ] Report bitsandbytes, MONAI, Transformers, and PEFT versions.
- [ ] Report free disk, model cache, dataset cache, and writable status.
- [ ] Avoid printing credentials, tokens, or patient-data paths.
- [ ] Provide machine-readable JSON output in addition to human-readable output.
- [ ] Add backend selection and report CUDA device count/NCCL availability or TPU device count/PJRT runtime.
- [ ] Report PyTorch-XLA/libtpu versions, TPU type/topology, XLA BF16 support, and SPMD availability.
- [ ] Report incompatible packages and custom CUDA extensions when TPU is selected.

### Containers and local parity

- [ ] Add development and CI Dockerfiles.
- [ ] Add a compose file with explicit GPU configuration and mounted cache locations.
- [ ] Run containers as a non-root user where practical.
- [ ] Pin the base image by digest or immutable version.
- [ ] Document host-driver compatibility and expected disk requirements.
- [ ] Add a CUDA image and a TPU VM/bootstrap environment; do not assume the CUDA image runs on TPU.
- [ ] Document Google Cloud TPU VM provisioning, storage, service-account, and network prerequisites without committing credentials.

### Reproducibility capture

- [ ] Implement a utility that captures commit SHA and dirty status.
- [ ] Capture lockfile hash, CUDA/driver versions, GPU model, seed, and precision.
- [ ] Define placeholders for future dataset, preprocessing, and base-model hashes.
- [ ] Capture trainable parameter counts, effective batch size, and peak VRAM.
- [ ] Serialize run metadata deterministically.
- [ ] Capture accelerator backend, topology, world size, compiler/runtime flags, precision policy, and shape-bucket set.
- [ ] Capture XLA compilation/metrics reports for TPU runs and NCCL/CUDA runtime details for distributed GPU runs.

### Local tracking

- [ ] Define a small tracker protocol.
- [ ] Implement a local JSON tracker first.
- [ ] Add TensorBoard support without making it mandatory.
- [ ] Reserve optional MLflow and W&B adapters behind extras.
- [ ] Redact configured sensitive keys before logging.

## Tests and verification

- [ ] Test a clean editable install in a fresh environment.
- [ ] Run imports on a CPU-only process.
- [ ] Run one synthetic MONAI 3D load/crop pipeline.
- [ ] Run one minimal LoRA optimization step using a tiny local model.
- [ ] On the target GPU, allocate BF16 tensors and report peak memory.
- [ ] Verify ignored data/model files are not considered for commit.
- [ ] Verify doctor JSON conforms to a schema.
- [ ] Run a one-step BF16 linear-model optimization on every local TPU device.
- [ ] Run a multi-device reduction test on CUDA and TPU protected runners.
- [ ] Verify CPU-only import does not initialize CUDA or XLA.
- [ ] Verify CUDA-only extensions are not imported by the TPU environment.

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

- [ ] CPU tests pass without downloading weights.
- [ ] Protected GPU smoke confirms BF16 execution.
- [ ] Protected TPU smoke confirms XLA BF16 execution on all local devices.
- [ ] Synthetic NIfTI preprocessing and tiny LoRA step pass.
- [ ] Runtime diagnostics are useful and redact sensitive values.
- [ ] No data, weights, caches, or credentials are tracked.

## Handoff

- [ ] Record exact supported Python/CUDA versions.
- [ ] Record optional dependencies unavailable on the target host.
- [ ] Record the tested CUDA and TPU runtime matrices and launch commands.
- [ ] Document package conventions for Phase 02.
- [ ] Provide the run-metadata API contract.
