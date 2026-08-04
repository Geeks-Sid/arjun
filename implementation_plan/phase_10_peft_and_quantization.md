# Phase 10: LoRA, QLoRA, and PEFT Subsystem

## Objective

Implement auditable, architecture-aware PEFT and quantization across language models, visual transformers, selected 3D encoders, and task modules.

CUDA and TPU do not share one quantization implementation. CUDA may use bitsandbytes QLoRA after capability checks. The baseline TPU path is BF16 LoRA/frozen-model adaptation; experimental XLA quantization must not be presented as accepted QLoRA.

## Dependencies

- [ ] Phases 06-09 expose model modules and capability metadata.
- [ ] Phase 05 loading modes and registry policies are stable.
- [ ] A tiny local transformer and representative real models are available for tests.

## Scope boundaries

Allowed areas: `medfm/peft/`, PEFT configuration schemas, module inspection CLI, adapter checkpoint utilities, and Phase 10 tests.

Do not implement task recipes or the generalized trainer loop.

## Implementation checklist

### Configuration and validation

- [ ] Implement versioned LoRA/QLoRA configuration matching `idea.md`.
- [ ] Support rank, alpha, dropout, bias, modules-to-save, adapter name, RS-LoRA, and DoRA flags.
- [ ] Support NF4, double quantization, and BF16 compute configuration.
- [ ] Reject incompatible dtypes, unsupported hardware, and unsupported model families early.
- [ ] Require explicit target confirmation for unknown architectures.
- [ ] Split quantization configuration from PEFT configuration so LoRA can run without quantization.
- [ ] Encode per-backend allowed combinations and fail before model allocation.

### Module resolver

- [ ] Build architecture-specific policies for supported LLMs and 2D/3D transformers.
- [ ] Resolve attention and optional MLP targets by module type and full path.
- [ ] Restrict 3D Swin defaults to reviewed late stages.
- [ ] Exclude patch embeddings, normalizations, convolutional stems, and full decoders by default.
- [ ] Implement `medfm peft inspect --model <id>` with name, type, shape, count, selected flag, and reason.
- [ ] Fail if configured target patterns match zero or unexpectedly broad modules.

### Injection and trainability

- [ ] Implement separate visual and language adapter injection.
- [ ] Support fully trainable bridges, new task heads, and new segmentation decoders alongside adapters.
- [ ] Support named adapters such as vision, language, task, site, and modality adapters.
- [ ] Set and verify `requires_grad` deterministically after every stage change.
- [ ] Produce a trainable-parameter audit before optimization.
- [ ] Inject adapters before distributed wrapping/sharding unless the selected backend documents another sequence.
- [ ] Verify LoRA parameters, modules-to-save, and tied embeddings remain visible after DDP/FSDP/XLA wrapping.

### Quantization safety

- [ ] Prepare quantized LLMs for k-bit training using supported APIs.
- [ ] Keep quantized base parameters out of the optimizer.
- [ ] Verify compute dtype and device placement.
- [ ] Disable training KV cache when required.
- [ ] Provide a typed capability error when bitsandbytes/CUDA is unavailable.
- [ ] Reject bitsandbytes for `xla_tpu` because TPU is not a documented supported bitsandbytes backend.
- [ ] Implement TPU BF16 LoRA and frozen-base adapter training as first-class configurations.
- [ ] Keep PyTorch/XLA quantized operations behind an `experimental_xla_quantization` flag and separate acceptance plan.
- [ ] Do not claim TPU QLoRA equivalence unless an upstream-supported 4-bit training implementation passes optimizer, parity, and checkpoint tests.

### Checkpointing and merge

- [ ] Save adapter-only checkpoints with base-model IDs/revisions and config hashes.
- [ ] Save bridge/head/decoder tensors separately in safetensors.
- [ ] Reload multiple named adapters without ambiguity.
- [ ] Support optional inference merge while preserving unmerged canonical artifacts.
- [ ] Compare merged and unmerged output within documented tolerance.
- [ ] Reject adapter loading against the wrong base revision or architecture.
- [ ] Export adapter weights as CPU safetensors independent of CUDA/XLA device state.
- [ ] Test CUDA-trained adapter load on TPU and TPU-trained adapter load on CUDA for a declared portable model.

## Tests and verification

- [ ] Attach LoRA to a tiny 2D ViT and verify selected gradients.
- [ ] Attach LoRA to a tiny 3D transformer and verify stage restriction.
- [ ] Attach QLoRA to the selected 4B LLM in the protected GPU environment.
- [ ] Verify total/frozen/adapter/bridge/head/decoder counts.
- [ ] Fail on zero trainable parameters.
- [ ] Fail when full LLM weights become trainable in QLoRA mode.
- [ ] Fail when quantized weights enter optimizer groups.
- [ ] Save/reload separate visual and language adapters.
- [ ] Pass merge equivalence and wrong-base rejection tests.
- [ ] Run BF16 LoRA optimizer steps on CUDA and TPU using identical tiny-model fixtures.
- [ ] Compare trainable module sets and update direction across backends.
- [ ] Run a real 4B QLoRA smoke on supported GPU hardware only.
- [ ] Verify TPU configuration reports BF16 LoRA rather than silently dropping quantization.
- [ ] Verify distributed wrapping preserves adapter state and gradient reduction.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)
- [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)
- [Transformers bitsandbytes hardware compatibility](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [PyTorch/XLA quantized operations, experimental](https://docs.pytorch.org/xla/master/perf/quantized_ops.html)

## Smoke command

```bash
python -m medfm.cli.peft inspect --model medsiglip_448
```

## Acceptance command

```bash
pytest tests/phase_10 -q && python -m medfm.tools.validate_phase --phase 10
```

## Exit criteria

- [ ] One 2D, one 3D, and one language adapter path pass gradient audits.
- [ ] QLoRA performs one optimizer step on the target GPU.
- [ ] BF16 LoRA performs one optimizer step on the target TPU topology.
- [ ] Separate adapters round-trip without base-weight duplication.
- [ ] Merge equivalence passes.
- [ ] Unknown or overbroad targeting fails closed.

## Handoff

- [ ] Publish optimizer-visible parameter groups for Phase 12.
- [ ] Publish adapter checkpoint schema and compatibility rules.
- [ ] Publish architecture target policies and measured trainable counts.
- [ ] Record unsupported quantization/model combinations.
- [ ] Publish the CUDA QLoRA versus TPU BF16 LoRA matrix and cross-backend adapter evidence.
