# Phase 12 handoff

## Recipe configuration schema

Use `RunConfig.from_dict`/`RunConfig.load` as the sole entry point. Keep recipe-specific model and dataset choices under `model`, `dataset`, `task`, `recipe`, and `extensions`; do not add model names or architecture branches to `Trainer`.

The canonical contract includes:

- `accelerator`: backend (`cpu`, `cuda`, `xla_tpu`), distribution, precision, attention, static-shape and compilation gates, rank/topology, and sharding mesh.
- `batch`: microbatch per device, accumulation, global-batch assertion, and final-batch policy.
- `memory`: reserved headroom, microbatch/token/patch limits, checkpointing, cache/frozen modes, and explicit offload.
- `optimizer`: named role groups, AdamW baseline, warmup/decay in optimizer steps, clipping, and explicitly gated fused/8-bit/sync-free modes.
- `freeze_schedule`: strictly increasing boundary steps ending in an open-ended stage.
- `peft`/`quantization`: LoRA/QLoRA policy validated before builders run.
- `checkpoint`, `tracking`, provenance hashes, and `extensions.shape_buckets`.

New recipe fields must be serializable, included in the canonical hash when scientific, and covered by a compatibility migration before changing `schema_version`.

## Known-safe smoke settings

- CPU: `configs/smoke/tiny_multitask.yaml`, FP32, one optimizer step, microbatch 2, accumulation 1.
- CUDA: the same config with `--backend cuda`; use only tiny/local models on this workstation and inspect the recorded CUDA memory snapshot.
- Dry run: add `--dry-run --format json`; require `allocated: false` before any weight-bearing recipe launch.
- Parity: `python -m medfm.cli.main accelerator parity --config configs/smoke/tiny_multitask.yaml --left cpu --right cuda --format json`.
- Profile: `python -m medfm.cli.main accelerator profile --config configs/smoke/tiny_multitask.yaml --backend cuda --format json`.
- TPU: require `PJRT_DEVICE=TPU`, static buckets, explicit topology, and a passing replicated acceptance marker before enabling SPMD/FSDP.

## Checkpoint/resume compatibility

A resumable checkpoint must retain the exact run-config hash, model id, base-model revision, backend, distribution, world size, topology, precision, static bucket schema, and sharding metadata. Resume rejects scientific configuration or topology changes by default. A topology change requires an explicit reshard workflow; `allow_topology_change=True` is not a substitute for resharding.

Single-device checkpoints use the atomic `model.pt`/optimizer/scheduler/scaler/RNG format. CUDA FSDP checkpoints use `torch.distributed.checkpoint` with a shared atomic staging directory; TPU SPMD checkpoints use `torch_xla.experimental.distributed_checkpoint.SPMDSavePlanner`/`SPMDLoadPlanner`. The DCP branch requires an initialized process group and every rank participates in shard I/O. Replicated TPU checkpoints are written only by rank 0 and synchronized before readers proceed.

`training_state.json` records global step, epoch, micro step, batch cursor, interruption state, metrics, best criterion, and freeze stage. `rng.pt`, optimizer/scheduler/scaler files, and sampler state are required whenever those objects exist. `adapter_only` artifacts cannot resume training; use `export_adapter` for portable deployment state.

## Protected GPU commands

Run only against an approved recipe/config and keep the command/config in the failure artifact:

```bash
python -m medfm.cli.train --config <approved.yaml> --dry-run --format json
python -m medfm.cli.train --config <approved.yaml> --backend cuda --format json
python -m medfm.cli.main accelerator profile --config <approved.yaml> --backend cuda --format json
python -m medfm.cli.main accelerator parity --config <approved.yaml> --left cpu --right cuda --format json
torchrun --standalone --nproc_per_node=<N> -m medfm.cli.train --config <approved.yaml> --backend cuda --format json
```

Do not auto-reduce batch size, token budgets, patch dimensions, precision, or distribution after OOM. Apply an ordered suggestion only by editing and reviewing the recipe explicitly.

## Protected TPU commands

```bash
PJRT_DEVICE=TPU python -m medfm.cli.train --config <approved_tpu.yaml> --dry-run --format json
PJRT_DEVICE=TPU python -m medfm.cli.train --config <approved_tpu.yaml> --backend xla_tpu --format json
PJRT_DEVICE=TPU python -m medfm.cli.main accelerator profile --config <approved_tpu.yaml> --backend xla_tpu --format json
PJRT_DEVICE=TPU python -m medfm.cli.main accelerator parity --config <approved_tpu.yaml> --left cpu --right xla_tpu --format json
```

First record a single-host replicated run with stable bucket compilation and portable adapter export. Only then enable the explicit `spmd_fsdp` distribution and an explicit sharding mesh. Preserve XLA metrics/profiles and coordinator-only replicated checkpoints.
