# Phase 12: Unified Training Engine and 48 GB Memory Planner

## Objective

Build one resumable trainer with CPU, CUDA, and PyTorch/XLA backends, task-specific step functions, explicit optimizer groups, staged freezing, reproducibility capture, compilation/memory diagnostics, and no silent resource-driven experiment mutation.

## Dependencies

- [x] Phases 09-11 provide models, bridges, PEFT, heads, and task modules.
- [x] Phase 01 tracking/reproducibility utilities are accepted.
- [x] Phase 04 collators can produce representative batches.
- [x] The mandatory [accelerator training strategy](accelerator_training_strategy.md) is accepted.

## Scope boundaries

Allowed areas: `medfm/training/`, trainer/config CLI, checkpoint manager, memory planner, tracking integration, and Phase 12 tests.

Do not encode recipe-specific model choices directly in the trainer.

## Implementation checklist

### Build pipeline

- [x] Implement typed `RunConfig` loading and canonical hashing.
- [x] Build registry, dataset, model, PEFT, optimizer, task, trainer, and evaluator in explicit stages.
- [x] Validate all capabilities before allocating large weights.
- [x] Keep `TrainingStep.forward_and_loss` task-specific behind a shared interface.
- [x] Make dry-run/model-summary mode available.
- [x] Implement `AcceleratorBackend` with CPU, CUDA, and XLA-TPU implementations.
- [x] Prohibit model/task code from reaching through the backend to CUDA/XLA-specific APIs.
- [x] Validate model/backend/precision/quantization/distribution compatibility before loading weights.

### Accelerate and precision

- [x] Use Accelerate for placement, mixed precision, accumulation, clipping, and future distributed compatibility.
- [x] Prefer BF16 where supported and keep sensitive losses/metrics FP32.
- [x] Keep NF4 as storage for QLoRA base weights only when the selected backend explicitly supports it.
- [x] Select SDPA first, FlashAttention when compatible, and eager fallback explicitly.
- [x] Disable training KV cache when required.
- [x] Use Accelerate for verified shared behavior while retaining direct XLA hooks for compile metrics, SPMD, and checkpoint planners.
- [x] Use CUDA BF16 where supported and FP16 plus gradient scaling only when explicitly selected.
- [x] Use XLA BF16 autocast on TPU without FP16-style gradient scaling.
- [x] Retie language-model weights after TPU placement when required.
- [x] Keep FP32 loss/metric accumulation consistent across backends.

### Launch and distributed execution

- [x] Support single-device CPU/CUDA launch.
- [x] Support CUDA DDP through `torchrun` or an accepted Accelerate launcher.
- [x] Support CUDA FSDP with transformer-block auto-wrap policies.
- [x] Support TPU PJRT replicated launch on all local devices.
- [x] Add TPU SPMD/FSDP only after replicated acceptance.
- [x] Record rank, local rank, host index, world size, topology, and sharding mesh.
- [x] Compute and log global batch from microbatch, world size, and accumulation.
- [x] Keep learning-rate scaling an explicit recipe decision when world size changes.
- [x] Synchronize failures, early stopping, and best-checkpoint decisions across ranks.

### Input and static-shape execution

- [x] Integrate distributed samplers with deterministic `set_epoch` behavior.
- [x] Use CUDA pinned-memory/nonblocking transfer where measured beneficial.
- [x] Use an XLA device loader/prefetch path and fixed-shape collator buckets on TPU.
- [x] Keep train and validation bucket sets bounded and warm them before measurement.
- [x] Ensure final TPU batches are padded/dropped according to config and masked from losses/metrics.
- [x] Detect shape-driven XLA recompilation and name the sample/bucket that caused it.

### Optimizers and staged freezing

- [x] Build separate bridge, task head, decoder, vision-LoRA, and language-LoRA parameter groups.
- [x] Validate every trainable parameter occurs exactly once.
- [x] Log group learning rates, counts, and weight decay.
- [x] Implement freeze schedules with boundary-step tests.
- [x] Rebuild or safely update optimizer groups when stages change.
- [x] Run gradient audits after construction and each stage transition.
- [x] Use AdamW as the portable baseline; gate fused, 8-bit, and XLA sync-free variants by backend evidence.
- [x] Define warmup/decay in optimizer steps and preserve schedule state across accumulation/resume.
- [x] Clip global gradient norm after unscaling where applicable and log norms per component.

### Memory planning

- [x] Reserve configurable runtime headroom rather than targeting all 48 GB.
- [x] Configure microbatch, accumulation, activation checkpointing, token limits, and patch sizes explicitly.
- [x] Estimate model, optimizer, gradient, activation, and input memory where possible.
- [x] Log allocated/reserved peak CUDA memory and configuration.
- [x] Produce an OOM diagnostic with ordered suggestions from `idea.md`.
- [x] Never silently retry with a modified scientific configuration.
- [x] Support explicit frozen embedding/token-cache training modes.
- [x] Keep CUDA VRAM and TPU HBM planners separate; never call CUDA memory APIs on XLA.
- [x] Add TPU recommendations in this order: fixed smaller bucket, fewer visual/text tokens, activation checkpointing, frozen/cached encoder, replicated-to-SPMD/FSDP transition.
- [x] Record XLA compile time, graph/compilation count, host-device transfers, and unsupported-op counters.
- [x] Fail the TPU performance gate on repeated steady-state recompilation above a configured threshold.

### Checkpointing and resume

- [x] Save adapter, bridge, head, decoder, optimizer, scheduler, scaler, step/epoch, and RNG states.
- [x] Save sampler/dataloader state where possible.
- [x] Save run config, hashes, base references, metrics, and best criterion.
- [x] Write checkpoints atomically and detect incomplete checkpoints.
- [x] Verify dataset/config/model compatibility on resume.
- [x] Support adapter-only deployment artifacts separately from resumable training checkpoints.
- [x] Use `torch.distributed.checkpoint` for sharded CUDA runs.
- [x] Use PyTorch/XLA SPMD save/load planners for sharded TPU runs.
- [x] Save replicated TPU state only from the master/coordinator and materialize portable exports on CPU.
- [x] Include backend, topology, world size, sharding, precision, compiler/runtime, and static bucket schema.
- [x] Support a canonical CPU/safetensors adapter export independent of resume checkpoint format.

### Tracking and failure handling

- [x] Log locally by default with sensitive-field redaction.
- [x] Record commit/dirty state, lockfile, dataset, preprocess, model, and configuration hashes.
- [x] Detect NaN/Inf loss and gradients with an actionable failure report.
- [x] Preserve the last safe checkpoint and command/config context on failure.
- [x] Make interruptions resumable without claiming phase success.
- [x] Keep `.item()`/host materialization out of the critical step except deliberate logging boundaries.
- [x] Save XLA metrics/profiles and CUDA memory/profiler summaries with run artifacts.
- [x] Detect NaN/Inf before distributed collectives can leave other ranks blocked.

## Tests and verification

- [x] Overfit one tiny batch for every task-step family.
- [x] Compare interrupted/resumed training to uninterrupted training.
- [x] Verify accumulation produces the expected optimizer-step count/effective batch size.
- [x] Verify gradient clipping and freeze schedule transitions.
- [x] Verify no frozen or quantized base parameter receives optimizer updates.
- [x] Simulate OOM and verify diagnostic output without automatic config mutation.
- [x] Corrupt a checkpoint and verify safe rejection.
- [x] Complete one classification, segmentation, 2D VLM, and 3D VLM optimizer step on target hardware.
- [x] Record and assert configurable peak-memory envelopes.
- [ ] Run identical tiny-model one-step parity tests on CPU, CUDA, and TPU (blocked: `torch_xla` is not installed; CPU/CUDA parity passed).
- [ ] Test CUDA DDP and TPU replicated reduction with the same global batch (blocked: only one CUDA device is available and `torch_xla` is not installed).
- [ ] Test one CUDA FSDP and one TPU SPMD/FSDP checkpoint/resume smoke where hardware is available (blocked: only one CUDA device is available and `torch_xla` is not installed).
- [ ] Verify repeated TPU steps in the same bucket do not recompile in steady state (blocked: `torch_xla` is not installed).
- [x] Verify padded distributed samples/tokens do not alter loss or metrics.
- [x] Verify topology changes either reshard correctly or fail before training.
- [ ] Measure compile time separately from TPU steady-state throughput (blocked: `torch_xla` is not installed).

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)
- [Accelerate TPU training](https://huggingface.co/docs/accelerate/basic_tutorials/tpu)
- [PyTorch/XLA SPMD](https://docs.pytorch.org/xla/master/spmd.html)
- [PyTorch/XLA mixed precision](https://docs.pytorch.org/xla/master/perf/amp.html)
- [PyTorch/XLA distributed checkpointing](https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html)
- [PyTorch distributed checkpointing](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [PyTorch/XLA profiling](https://docs.pytorch.org/xla/master/learn/xla-profiling.html)

## Smoke command

```bash
python -m medfm.cli.train --config configs/smoke/tiny_multitask.yaml
```

## Acceptance command

```bash
pytest tests/phase_12 -q && python -m medfm.tools.validate_phase --phase 12
```

## Exit criteria

- [x] Core task families each complete an optimizer step.
- [x] Resume equivalence passes within documented tolerance.
- [x] Gradient and optimizer audits pass.
- [x] Peak VRAM and full run provenance are recorded.
- [x] OOM behavior is diagnostic and scientifically explicit.
- [ ] One CUDA and one TPU optimizer step pass for every baseline task family using tiny/local models (blocked: CUDA baseline passes; TPU steps require `torch_xla`).
- [ ] TPU acceptance includes stable compilation and portable adapter export (blocked: requires `torch_xla`/PJRT).

## Handoff

- [x] Publish recipe configuration schema and extension points.
- [x] Publish memory profiles and known-safe smoke settings.
- [x] Publish checkpoint/resume compatibility rules.
- [x] Record protected GPU commands for recipe phases.
- [x] Record protected TPU launch, profiling, checkpoint, and parity commands for recipe phases.
