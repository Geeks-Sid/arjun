# transfer_plan/peft/lora.md

Source: `medfm/peft/lora.py` (~740 lines: custom LoRA injection, named adapters, merge/unmerge,
audits).

Wave: 0 — **high caution**. Per the ground rules, trainer-adjacent libraries are adopted only
when the library is very mature AND clearly valuable. `peft` (0.20.0, verified installed:
`LoraConfig`, `get_peft_model`, `PeftModel`) is mature — but this module exists precisely to be
**backend-neutral** (CPU / CUDA / PyTorch-XLA TPU), which upstream `peft` does not certify for
the XLA path, and it carries a contract (named adapters, resolver policy, checkpoint schema,
quantization gating) that peft's model would not preserve without a rewrite.

## Transfer checklist

- [ ] `LoRALinear` low-rank math (`lora_A @ lora_B` scaled, dropout, optional DoRA magnitude)
      → **partial / evaluate-first** — math is identical to peft's `LoraLayer` forward. A
      candidate adoption is using `peft.LoraConfig` + `peft.get_peft_model` **only for the
      rank/alpha/dropout/rslora/fan_in_fan_out hyperparameters and the layer math**. **Must**
      be gated by: (1) a parity test on the TPU path is impossible here, so compare CUDA/CPU
      numerically and keep our XLA path untouched; (2) checkpoint schema (safetensors +
      config_hash) must stay ours; (3) the resolver/audit surface (inject_lora, add_named_adapter,
      set_active_adapter, merge/unmerge) is contract and stays custom. Expected verdict:
      **keep with a documented parity harness**, unless a later phase decides to adopt
      `peft.PeftModel` wholesale — which is an architectural decision beyond this plan.
- [ ] `inject_lora` / `inject_visual_lora` / `inject_language_lora` target resolution +
      module matching (`resolve_targets`, pattern safety) → **keep** — the `TargetPolicy`
      resolver is bespoke and must not be replaced by peft's `target_modules` conventions
      (they differ in pattern safety + architecture scoping).
- [ ] `merge_lora_adapters` / `unmerge_lora_adapters` → **keep** — merge state machine specific
      to our named-adapters; peft's `merge_and_unload` semantics differ (unloads base, one
      active adapter).
- [ ] `audit_trainable_parameters` / `configure_trainability` / `verify_adapter_state` →
      **keep** — QLoRA-safety audits and freeze-order determinism are contract.
- [ ] `is_quantized_parameter` / `_is_adapter_name` → **keep** (helpers).

## Tests
`tests/phase_06/test_lora.py`, `tests/phase_06/test_checkpoint.py`,
`tests/phase_10/test_resolver_and_injection.py`, `tests/phase_10/test_quantization_safety.py`,
`tests/phase_10/test_checkpoint_and_merge.py`.
