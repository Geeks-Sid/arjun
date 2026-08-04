# Containers and local parity

## Images

- `docker/Dockerfile` — CUDA development image (`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`,
  immutable version tag; pin by digest for release builds). Runs as the
  non-root `medfm` user.
- `docker/Dockerfile.ci` — CPU CI image (`python:3.13-slim-bookworm`) running
  `make lint && make typecheck && make test && make smoke`.
- `docker/compose.yaml` — dev service with an explicit NVIDIA device
  reservation and named volumes for the model cache, dataset cache, and
  artifacts.

```bash
docker compose -f docker/compose.yaml build dev
docker compose -f docker/compose.yaml run --rm dev make test
docker compose -f docker/compose.yaml run --rm dev make test-gpu
```

## Host-driver compatibility

The pinned PyTorch build (torch 2.9.0, CUDA 12.8 wheels from PyPI) requires a
host NVIDIA driver >= 525.60.13 (CUDA 12.0 minimum); the 48 GB reference GPU
and any Ada/Hopper device need driver >= 545 for full BF16 support. Verify
with `nvidia-smi` on the host — the container uses the host driver, never its
own. Expected disk: ~15 GB for the dev image, ~8 GB for the CI image, plus
model/dataset cache volumes (budget 100+ GB on training hosts).

## TPU VM provisioning (Google Cloud)

The CUDA image does **not** run on TPU. Use a TPU VM with the repo's `tpu`
extra instead:

1. Create a queued resource or on-demand TPU VM with a `tpu-ubuntu2204-base`
   (or newer) image, e.g.:
   `gcloud compute tpus tpu-vm create <name> --zone=<zone> --accelerator-type=v4-8 --version=tpu-ubuntu2204-base`
2. Attach a service account with read access to the project's storage buckets
   (`roles/storage.objectViewer` minimum). Never commit service-account keys;
   use VM-attached service accounts only.
3. Storage: persistent disk >= 200 GB for caches; datasets should live in
   regional GCS buckets co-located with the TPU zone.
4. Network: egress to `pypi.org` (or an internal mirror) for the bootstrap;
   private Google access for GCS.
5. Bootstrap: `REPO_DIR=$HOME/arjun bash scripts/tpu_vm_bootstrap.sh`
   (installs uv, syncs the `tpu` extra, runs the TPU doctor and smoke).

The TPU baseline is torch==2.9.0 + torch_xla[tpu]==2.9.0 (libtpu 0.0.21).
bitsandbytes, FlashAttention, and cuCIM are CUDA-only and must never be
installed on the TPU image; `medfm doctor --backend xla_tpu` reports them as
incompatible packages if present.
