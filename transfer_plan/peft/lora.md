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

- [x] `LoRALinear` low-rank math (`lora_A @ lora_B` scaled, dropout, optional DoRA magnitude)
      → **keep with documented parity harness** — custom math remains the backend-neutral path;
      CPU/CUDA numerical parity is an optional adoption gate, but no `peft.get_peft_model`
      adoption is justified while TPU/XLA behavior and our schema/audit contracts differ.
- [x] `inject_lora` / `inject_visual_lora` / `inject_language_lora` target resolution +
      module matching (`resolve_targets`, pattern safety) → **keep** — the `TargetPolicy`
      resolver is bespoke and must not be replaced by peft's `target_modules` conventions
      (they differ in pattern safety + architecture scoping).
- [x] `merge_lora_adapters` / `unmerge_lora_adapters` → **keep** — merge state machine specific
      to our named-adapters; peft's `merge_and_unload` semantics differ (unloads base, one
      active adapter).
- [x] `audit_trainable_parameters` / `configure_trainability` / `verify_adapter_state` →
      **keep** — QLoRA-safety audits and freeze-order determinism are contract.
- [x] `is_quantized_parameter` / `_is_adapter_name` → **keep** (helpers).

## Tests
`tests/phase_06/test_lora.py`, `tests/phase_06/test_checkpoint.py`,
`tests/phase_10/test_resolver_and_injection.py`, `tests/phase_10/test_quantization_safety.py`,
`tests/phase_10/test_checkpoint_and_merge.py`.

## Result

Verified keep (with documented parity-harness decision) for all five items. `peft` 0.20.0
symbols are installed, but `peft.get_peft_model` is not adopted: custom named-adapter,
resolver, checkpoint/config-hash, quantization-audit, merge/unmerge, and XLA-safe contracts
remain authoritative. The optional numerical CPU/CUDA peft parity harness was intentionally
not added because no library transfer is being pursued in this checklist. Tests green:
`uv run --frozen pytest tests/phase_06/test_lora.py tests/phase_06/test_checkpoint.py`
(15 passed) and `uv run --frozen pytest tests/phase_10/test_resolver_and_injection.py
tests/phase_10/test_quantization_safety.py tests/phase_10/test_checkpoint_and_merge.py`
(12 passed). No source/test files were modified.
