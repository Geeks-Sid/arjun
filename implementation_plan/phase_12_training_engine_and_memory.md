# Phase 12: Unified Training Engine and 48 GB Memory Planner

## Objective

Build one resumable trainer with CPU, CUDA, and PyTorch/XLA backends, task-specific step functions, explicit optimizer groups, staged freezing, reproducibility capture, compilation/memory diagnostics, and no silent resource-driven experiment mutation.

## Dependencies

- [ ] Phases 09-11 provide models, bridges, PEFT, heads, and task modules.
- [ ] Phase 01 tracking/reproducibility utilities are accepted.
- [ ] Phase 04 collators can produce representative batches.
- [ ] The mandatory [accelerator training strategy](accelerator_training_strategy.md) is accepted.

## Scope boundaries

Allowed areas: `medfm/training/`, trainer/config CLI, checkpoint manager, memory planner, tracking integration, and Phase 12 tests.

Do not encode recipe-specific model choices directly in the trainer.

## Implementation checklist

### Build pipeline

- [ ] Implement typed `RunConfig` loading and canonical hashing.
- [ ] Build registry, dataset, model, PEFT, optimizer, task, trainer, and evaluator in explicit stages.
- [ ] Validate all capabilities before allocating large weights.
- [ ] Keep `TrainingStep.forward_and_loss` task-specific behind a shared interface.
- [ ] Make dry-run/model-summary mode available.
- [ ] Implement `AcceleratorBackend` with CPU, CUDA, and XLA-TPU implementations.
- [ ] Prohibit model/task code from reaching through the backend to CUDA/XLA-specific APIs.
- [ ] Validate model/backend/precision/quantization/distribution compatibility before loading weights.

### Accelerate and precision

- [ ] Use Accelerate for placement, mixed precision, accumulation, clipping, and future distributed compatibility.
- [ ] Prefer BF16 where supported and keep sensitive losses/metrics FP32.
- [ ] Keep NF4 as storage for QLoRA base weights only when the selected backend explicitly supports it.
- [ ] Select SDPA first, FlashAttention when compatible, and eager fallback explicitly.
- [ ] Disable training KV cache when required.
- [ ] Use Accelerate for verified shared behavior while retaining direct XLA hooks for compile metrics, SPMD, and checkpoint planners.
- [ ] Use CUDA BF16 where supported and FP16 plus gradient scaling only when explicitly selected.
- [ ] Use XLA BF16 autocast on TPU without FP16-style gradient scaling.
- [ ] Retie language-model weights after TPU placement when required.
- [ ] Keep FP32 loss/metric accumulation consistent across backends.

### Launch and distributed execution

- [ ] Support single-device CPU/CUDA launch.
- [ ] Support CUDA DDP through `torchrun` or an accepted Accelerate launcher.
- [ ] Support CUDA FSDP with transformer-block auto-wrap policies.
- [ ] Support TPU PJRT replicated launch on all local devices.
- [ ] Add TPU SPMD/FSDP only after replicated acceptance.
- [ ] Record rank, local rank, host index, world size, topology, and sharding mesh.
- [ ] Compute and log global batch from microbatch, world size, and accumulation.
- [ ] Keep learning-rate scaling an explicit recipe decision when world size changes.
- [ ] Synchronize failures, early stopping, and best-checkpoint decisions across ranks.

### Input and static-shape execution

- [ ] Integrate distributed samplers with deterministic `set_epoch` behavior.
- [ ] Use CUDA pinned-memory/nonblocking transfer where measured beneficial.
- [ ] Use an XLA device loader/prefetch path and fixed-shape collator buckets on TPU.
- [ ] Keep train and validation bucket sets bounded and warm them before measurement.
- [ ] Ensure final TPU batches are padded/dropped according to config and masked from losses/metrics.
- [ ] Detect shape-driven XLA recompilation and name the sample/bucket that caused it.

### Optimizers and staged freezing

- [ ] Build separate bridge, task head, decoder, vision-LoRA, and language-LoRA parameter groups.
- [ ] Validate every trainable parameter occurs exactly once.
- [ ] Log group learning rates, counts, and weight decay.
- [ ] Implement freeze schedules with boundary-step tests.
- [ ] Rebuild or safely update optimizer groups when stages change.
- [ ] Run gradient audits after construction and each stage transition.
- [ ] Use AdamW as the portable baseline; gate fused, 8-bit, and XLA sync-free variants by backend evidence.
- [ ] Define warmup/decay in optimizer steps and preserve schedule state across accumulation/resume.
- [ ] Clip global gradient norm after unscaling where applicable and log norms per component.

### Memory planning

- [ ] Reserve configurable runtime headroom rather than targeting all 48 GB.
- [ ] Configure microbatch, accumulation, activation checkpointing, token limits, and patch sizes explicitly.
- [ ] Estimate model, optimizer, gradient, activation, and input memory where possible.
- [ ] Log allocated/reserved peak CUDA memory and configuration.
- [ ] Produce an OOM diagnostic with ordered suggestions from `idea.md`.
- [ ] Never silently retry with a modified scientific configuration.
- [ ] Support explicit frozen embedding/token-cache training modes.
- [ ] Keep CUDA VRAM and TPU HBM planners separate; never call CUDA memory APIs on XLA.
- [ ] Add TPU recommendations in this order: fixed smaller bucket, fewer visual/text tokens, activation checkpointing, frozen/cached encoder, replicated-to-SPMD/FSDP transition.
- [ ] Record XLA compile time, graph/compilation count, host-device transfers, and unsupported-op counters.
- [ ] Fail the TPU performance gate on repeated steady-state recompilation above a configured threshold.

### Checkpointing and resume

- [ ] Save adapter, bridge, head, decoder, optimizer, scheduler, scaler, step/epoch, and RNG states.
- [ ] Save sampler/dataloader state where possible.
- [ ] Save run config, hashes, base references, metrics, and best criterion.
- [ ] Write checkpoints atomically and detect incomplete checkpoints.
- [ ] Verify dataset/config/model compatibility on resume.
- [ ] Support adapter-only deployment artifacts separately from resumable training checkpoints.
- [ ] Use `torch.distributed.checkpoint` for sharded CUDA runs.
- [ ] Use PyTorch/XLA SPMD save/load planners for sharded TPU runs.
- [ ] Save replicated TPU state only from the master/coordinator and materialize portable exports on CPU.
- [ ] Include backend, topology, world size, sharding, precision, compiler/runtime, and static bucket schema.
- [ ] Support a canonical CPU/safetensors adapter export independent of resume checkpoint format.

### Tracking and failure handling

- [ ] Log locally by default with sensitive-field redaction.
- [ ] Record commit/dirty state, lockfile, dataset, preprocess, model, and configuration hashes.
- [ ] Detect NaN/Inf loss and gradients with an actionable failure report.
- [ ] Preserve the last safe checkpoint and command/config context on failure.
- [ ] Make interruptions resumable without claiming phase success.
- [ ] Keep `.item()`/host materialization out of the critical step except deliberate logging boundaries.
- [ ] Save XLA metrics/profiles and CUDA memory/profiler summaries with run artifacts.
- [ ] Detect NaN/Inf before distributed collectives can leave other ranks blocked.

## Tests and verification

- [ ] Overfit one tiny batch for every task-step family.
- [ ] Compare interrupted/resumed training to uninterrupted training.
- [ ] Verify accumulation produces the expected optimizer-step count/effective batch size.
- [ ] Verify gradient clipping and freeze schedule transitions.
- [ ] Verify no frozen or quantized base parameter receives optimizer updates.
- [ ] Simulate OOM and verify diagnostic output without automatic config mutation.
- [ ] Corrupt a checkpoint and verify safe rejection.
- [ ] Complete one classification, segmentation, 2D VLM, and 3D VLM optimizer step on target hardware.
- [ ] Record and assert configurable peak-memory envelopes.
- [ ] Run identical tiny-model one-step parity tests on CPU, CUDA, and TPU.
- [ ] Test CUDA DDP and TPU replicated reduction with the same global batch.
- [ ] Test one CUDA FSDP and one TPU SPMD/FSDP checkpoint/resume smoke where hardware is available.
- [ ] Verify repeated TPU steps in the same bucket do not recompile in steady state.
- [ ] Verify padded distributed samples/tokens do not alter loss or metrics.
- [ ] Verify topology changes either reshard correctly or fail before training.
- [ ] Measure compile time separately from TPU steady-state throughput.

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

- [ ] Core task families each complete an optimizer step.
- [ ] Resume equivalence passes within documented tolerance.
- [ ] Gradient and optimizer audits pass.
- [ ] Peak VRAM and full run provenance are recorded.
- [ ] OOM behavior is diagnostic and scientifically explicit.
- [ ] One CUDA and one TPU optimizer step pass for every baseline task family using tiny/local models.
- [ ] TPU acceptance includes stable compilation and portable adapter export.

## Handoff

- [ ] Publish recipe configuration schema and extension points.
- [ ] Publish memory profiles and known-safe smoke settings.
- [ ] Publish checkpoint/resume compatibility rules.
- [ ] Record protected GPU commands for recipe phases.
- [ ] Record protected TPU launch, profiling, checkpoint, and parity commands for recipe phases.
