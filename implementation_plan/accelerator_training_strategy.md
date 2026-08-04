# Accelerator Training Strategy

This document is the mandatory cross-phase contract for making the framework trainable on CPUs for tests, NVIDIA GPUs for development/production training, and Google Cloud TPUs through PyTorch/XLA. It does not promise that every third-party medical model works on every accelerator. Compatibility must be measured and recorded per model, loading mode, task, and topology.

## Supported execution tiers

- [ ] Tier 0: CPU contract tests with tiny local models and synthetic data.
- [ ] Tier 1: one CUDA GPU, initially a 48 GB device.
- [ ] Tier 2: multiple CUDA GPUs using DDP for replicated models.
- [ ] Tier 3: multiple CUDA GPUs using FSDP when model/optimizer state must be sharded.
- [ ] Tier 4: one TPU VM/slice using PyTorch/XLA PJRT with all local TPU devices.
- [ ] Tier 5: multi-host TPU SPMD/FSDP is supported only after single-host TPU acceptance.
- [ ] Record accelerator generation, device count, per-device memory, topology, runtime, compiler, and host resources for every run.

## Backend architecture

- [ ] Introduce an `AcceleratorBackend` interface owned by the training layer, not by models or tasks.
- [ ] Implement `CpuBackend`, `CudaBackend`, and `XlaTpuBackend`.
- [ ] Let Hugging Face Accelerate handle common placement, accumulation, mixed precision, and reductions where its behavior is verified.
- [ ] Use direct PyTorch/XLA APIs behind `XlaTpuBackend` for compilation metrics, SPMD sharding, XLA-specific checkpointing, and step synchronization.
- [ ] Prohibit application code from calling `.cuda()`, `torch.cuda.current_device()`, or constructing tensors on a hard-coded device.
- [ ] Construct tensors from an existing tensor or a backend-provided device/dtype factory.
- [ ] Keep model, task, loss, and metric APIs ordinary PyTorch so they remain backend-neutral.
- [ ] Isolate CUDA-only and XLA-only imports inside backend/capability modules.

## Capability matrix

Every model registry entry must record one status per backend:

```text
UNTESTED
CPU_CONTRACT_ONLY
SUPPORTED_SINGLE_DEVICE
SUPPORTED_REPLICATED
SUPPORTED_SHARDED
BLOCKED_CUSTOM_OP
BLOCKED_MEMORY
BLOCKED_UPSTREAM
NOT_APPLICABLE
```

- [ ] Record tested PyTorch, CUDA or PyTorch/XLA, Transformers, PEFT, and model revisions.
- [ ] Record required custom kernels and whether a pure-PyTorch/SDPA fallback exists.
- [ ] Record supported precision, quantization, attention implementation, and distributed mode.
- [ ] Record the exact smoke config and last successful date.
- [ ] Make TPU eligibility a registry validation rule rather than an assumption based on a CUDA pass.

## Precision and quantization policy

### CUDA

- [ ] Prefer BF16 on hardware with native BF16 support.
- [ ] Permit FP16 with gradient scaling only when BF16 is unavailable or a model requires it.
- [ ] Keep numerically sensitive reductions, losses, and metrics in FP32.
- [ ] Permit bitsandbytes NF4 QLoRA only on a backend explicitly supported by the installed bitsandbytes build.
- [ ] Audit that quantized base weights are excluded from optimizer state.
- [ ] Support SDPA by default and optional FlashAttention only when the pinned model and kernel pass parity tests.

### TPU

- [ ] Use XLA BF16 autocast for forward and loss computation; do not use FP16-style gradient scaling.
- [ ] Keep selected losses, accumulators, calibration, and metrics in FP32.
- [ ] Use BF16 LoRA, frozen encoders, bridge/head/decoder training, and sharding as the initial TPU strategy.
- [ ] Do not label bitsandbytes NF4 QLoRA as TPU-supported; bitsandbytes does not list TPU as a supported backend.
- [ ] Treat PyTorch/XLA quantized operations as experimental and outside the baseline until a dedicated parity/performance phase accepts them.
- [ ] Record XLA matmul precision settings and compare accuracy against the CUDA/FP32 reference.

## Static-shape and compilation policy for TPU

- [ ] Bucket 2D image dimensions, 3D patch shapes, number of images/slices, WSI tile counts, visual-token counts, and text lengths.
- [ ] Pad each bucket to fixed shapes and provide masks; never let sample content change control flow in a compiled training step.
- [ ] Use a bounded number of documented buckets to avoid repeated XLA compilation.
- [ ] Set `drop_last` or pad the final distributed training batch so per-replica shapes remain stable.
- [ ] Keep training and validation shape sets separate and precompile/warm each intended bucket.
- [ ] Avoid Python loops whose iteration count depends on batch data inside model forward.
- [ ] Replace unsupported/dynamic operations with tested static equivalents or mark the model/task TPU-blocked.
- [ ] Count XLA compilations and fail the TPU performance gate when steady-state recompilation exceeds the configured threshold.
- [ ] Save PyTorch/XLA metrics reports and profiler traces in run artifacts.

## Input pipeline

- [ ] Keep DICOM, NIfTI, WSI decoding, deterministic preprocessing, and heavy augmentation on CPU hosts unless a tested accelerator implementation exists.
- [ ] Materialize fixed-shape tensors before device transfer.
- [ ] Use distributed samplers with deterministic epoch seeds and no cross-rank sample duplication beyond explicit padding.
- [ ] Use backend-aware prefetch/device loaders for TPU and pinned-memory/nonblocking transfer for CUDA where beneficial.
- [ ] Tune worker count, prefetch depth, persistent workers, and host RAM independently by backend.
- [ ] Shard WSI embedding reads and volume samples across ranks before transfer.
- [ ] Record input wait time, host-to-device time, examples/sec, and accelerator utilization.
- [ ] Ensure one slow/corrupt sample is handled before collective synchronization to avoid rank hangs.

## Distributed semantics

- [ ] Define microbatch per device, world size, accumulation steps, and global batch explicitly.
- [ ] Record the formula `global_batch = microbatch_per_device * world_size * accumulation_steps`.
- [ ] Scale learning rate only through an explicit recipe option; never silently scale after topology changes.
- [ ] Reduce losses by the true supervised-example/token count, not a simple mean of uneven local means.
- [ ] Gather patient/study/slide IDs safely for evaluation without duplicating padded samples.
- [ ] Synchronize early stopping, best-checkpoint decisions, and failure status across ranks.
- [ ] Seed model initialization consistently and derive rank/worker data seeds deterministically.
- [ ] Verify tied language-model weights after accelerator placement, especially on TPU.

## Parallelism selection

- [ ] Use single-device training when the model and optimizer fit with required headroom.
- [ ] Use DDP/XLA replicated data parallelism when each replica fits and throughput is the goal.
- [ ] Use FSDP on CUDA when base model, gradients, or optimizer states do not fit per GPU.
- [ ] Use PyTorch/XLA SPMD/FSDP only after a replicated TPU baseline is correct.
- [ ] Define auto-wrap policies around transformer blocks, not arbitrary leaf modules.
- [ ] Apply activation checkpointing before the XLA FSDP wrapper where required by PyTorch/XLA.
- [ ] Validate that LoRA modules and modules-to-save are sharded/replicated as intended.
- [ ] Save and visualize sharding specifications for accepted TPU SPMD runs.

## Training stability techniques

- [ ] Require a one-batch overfit on each backend before long runs.
- [ ] Use AdamW as the baseline optimizer; introduce fused, 8-bit, or sync-free variants only behind backend capability checks.
- [ ] Use explicit warmup and decay schedules measured in optimizer steps, not raw dataloader iterations.
- [ ] Clip gradients by global norm after unscaling where scaling applies.
- [ ] Log pre-clip and post-clip norms by trainable component.
- [ ] Check loss and gradients for NaN/Inf before collectives can deadlock.
- [ ] Use activation checkpointing selectively and test dropout/RNG equivalence.
- [ ] Keep KV cache disabled during language-model training.
- [ ] Start from frozen encoder/head baselines before LoRA, and from LoRA before broader unfreezing.
- [ ] Add gradient accumulation only after a one-step non-accumulated reference test.

## Checkpoint portability

- [ ] Keep canonical deployment tensors accelerator-neutral in CPU safetensors.
- [ ] Keep resumable distributed checkpoints separate from portable adapter exports.
- [ ] Use `torch.distributed.checkpoint` for sharded CUDA runs.
- [ ] Use PyTorch/XLA SPMD checkpoint planners for sharded TPU runs.
- [ ] Save on the designated coordinator/master only when state is replicated.
- [ ] Include world size, topology, sharding, precision, compiler/runtime, and optimizer schema.
- [ ] Test same-backend resume equivalence.
- [ ] Test adapter export on CUDA, load on CPU, and load on TPU for models declared cross-backend.
- [ ] Test resharding when supported; otherwise reject topology changes with an actionable error.

## Backend parity tests

- [ ] Compare CPU FP32, CUDA BF16/FP32, and TPU BF16 forward outputs on deterministic tiny fixtures.
- [ ] Compare one optimizer step using identical initial weights and batches within declared tolerances.
- [ ] Compare loss masks, trainable parameters, gradient presence, and update direction.
- [ ] Compare checkpoint reload outputs.
- [ ] Compare metric aggregation with uneven and padded distributed batches.
- [ ] Set task-specific tolerances; do not demand bitwise equality across accelerators.
- [ ] Investigate divergence before accepting backend-specific thresholds.

## Performance acceptance

### CUDA

- [ ] Record allocated/reserved peak VRAM, examples/sec, tokens/sec, data wait, and utilization.
- [ ] Run a warmup before measuring steady-state throughput.
- [ ] Verify no accidental CPU offload or host synchronization in the critical step.
- [ ] Preserve at least the configured VRAM headroom.

### TPU

- [ ] Record compile time separately from steady-state step time.
- [ ] Record compilation count, graph count, device-host transfers, and XLA fallback counters.
- [ ] Fail acceptance on repeated steady-state compilation or unsupported-op CPU fallback above threshold.
- [ ] Profile at least one representative run and inspect host input stalls.
- [ ] Measure examples/sec per chip and scaling efficiency across the target slice.

## Required commands

Implement commands equivalent to:

```bash
medfm doctor --backend cuda
medfm doctor --backend xla_tpu
medfm train --config <config> --backend cuda
medfm train --config <config> --backend xla_tpu
medfm accelerator validate-model --model <id> --backend <backend>
medfm accelerator parity --config <tiny-config> --left cpu --right cuda
medfm accelerator parity --config <tiny-config> --left cuda --right xla_tpu
medfm accelerator profile --config <config> --backend <backend>
```

## Canonical backend configuration

The final schema may evolve, but implementation must preserve these distinctions:

```yaml
accelerator:
  backend: cuda
  distribution: single  # single | ddp | fsdp
  precision: bf16
  world_size: 1
  compile: false
  attention: sdpa
  static_shapes: false

quantization:
  method: bitsandbytes_nf4
  enabled: true

batch:
  microbatch_per_device: 1
  gradient_accumulation_steps: 16
  global_batch_size: 16
```

```yaml
accelerator:
  backend: xla_tpu
  distribution: replicated  # replicated | spmd_fsdp
  precision: bf16
  world_size: 8
  attention: xla
  static_shapes: true
  fail_on_recompilation_after_warmup: true
  max_steady_state_compilations: 0

quantization:
  enabled: false

batch:
  microbatch_per_device: 1
  gradient_accumulation_steps: 2
  global_batch_size: 16

shape_buckets:
  image_2d: [[448, 448]]
  volume_3d: [[96, 128, 128]]
  images_per_sample: [1, 8, 16]
  wsi_tiles: [256, 512, 1024]
  visual_tokens: [32, 64, 128]
  text_tokens: [256, 512, 1024]
```

- [ ] Validate configured global batch against microbatch, world size, and accumulation.
- [ ] Validate TPU configs use only declared shape buckets.
- [ ] Validate quantization against backend capability before model loading.
- [ ] Record resolved backend configuration after launcher/runtime discovery.
- [ ] Hash the resolved accelerator, distribution, precision, and bucket configuration into every run.

## Baseline acceptance matrix

- [ ] CPU: all schema, data, loss, metric, checkpoint-format, and tiny-model contract tests.
- [ ] CUDA single-device: one 2D classifier, one 3D classifier, one segmentation model, one WSI aggregator, and one VLM optimizer step.
- [ ] CUDA distributed: one DDP recipe and one FSDP/resharding smoke where hardware is available.
- [ ] TPU single-host: the same task families with tiny/local models, plus at least one real supported HF vision or language backbone.
- [ ] TPU model roster: every v1 model is marked supported, blocked, or untested with evidence; no blanket TPU claim is allowed.
- [ ] Cross-backend: portable adapter artifact loads on CPU, CUDA, and TPU for at least one declared portable model.

## Primary references

- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/master/)
- [Migrating PyTorch GPU code to PyTorch/XLA on TPU](https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html)
- [PyTorch/XLA SPMD guide](https://docs.pytorch.org/xla/master/spmd.html)
- [PyTorch/XLA distributed checkpointing](https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html)
- [PyTorch/XLA automatic mixed precision](https://docs.pytorch.org/xla/master/perf/amp.html)
- [PyTorch/XLA troubleshooting and compilation metrics](https://docs.pytorch.org/xla/master/debug.html)
- [PyTorch/XLA profiling](https://docs.pytorch.org/xla/master/learn/xla-profiling.html)
- [Hugging Face Accelerate TPU guide](https://huggingface.co/docs/accelerate/basic_tutorials/tpu)
- [Hugging Face Accelerate TPU best practices](https://huggingface.co/docs/accelerate/concept_guides/training_tpu)
- [Transformers bitsandbytes hardware compatibility](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [PyTorch distributed checkpointing](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [Cloud TPU introduction and shape guidance](https://cloud.google.com/tpu/docs/intro-to-tpu)
