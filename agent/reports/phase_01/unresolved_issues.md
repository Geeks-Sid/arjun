# Phase 01 — Unresolved issues

1. **TPU hardware verification not run on this workstation.** There is no TPU
   attached to the development host. The TPU tests (`tests/phase_01/test_tpu.py`:
   one-step BF16 optimization on every local device, multi-device all-reduce) are
   implemented and protected behind `MEDFM_RUN_TPU_TESTS=1`; they must be executed
   on a Cloud TPU VM via `scripts/tpu_vm_bootstrap.sh` / `make test-tpu` before any
   TPU support claim is recorded in the model registry (Phase 05+ enforces this).
2. **Multi-GPU distributed tests not run.** The workstation has a single GPU;
   `test_multi_device_reduction` skips with a recorded reason. Run
   `make test-distributed-gpu` on a multi-GPU runner.
3. **Reference GPU mismatch.** Phase 00 targets a 48 GB VRAM reference device; this
   host has an 8 GB RTX 4060 Laptop GPU. BF16 behavior is verified, but
   memory-related acceptance (e.g. 48 GB headroom policies) needs the reference
   hardware.
4. **Containers not built locally.** Dockerfiles and compose are provided and
   statically reviewed; no local `docker build` was run in this phase (docker
   daemon availability on the host was not assumed). Build parity should be
   exercised in CI (Phase 18 hardening at the latest).
5. **TPU extra not installed on this host.** The `tpu` lock resolution was audited
   (torch 2.9.0 + torch-xla 2.9.0 + libtpu 0.0.21, no CUDA-only packages) but the
   runtime itself is only exercisable on a TPU VM.
