# Phase 10 -> Phase 11/12 handoff

## Optimizer-visible contract

Use `medfm.peft.optimizer_parameter_groups(model, learning_rates=..., weight_decay=...)` after adapter injection and stage freezing. Each group contains `name`, `params`, `parameter_names`, `lr`, and `weight_decay`; groups are disjoint and quantized base parameters are rejected. `audit_trainable_parameters(model, qlora=True)` is the required pre-optimizer gate.

## Adapter injection contract

- Visual adapters: `inject_visual_lora(adapter, LoRAConfig(...))` targets `adapter.backbone` and honors the registered architecture policy.
- Language adapters: `inject_language_lora(adapter, LoRAConfig(...))` targets `adapter.model`, leaving bridges/boundary components available for explicit training.
- Generic models: `inject_lora(model, config, architecture=...)`.
- Named adapters: set `adapter_name` to `vision`, `language`, `task`, `site`, or `modality`; use `add_named_adapter` for a second adapter and `set_active_adapter` to select it.
- Run `verify_adapter_state` after DDP/FSDP/XLA wrapping. Adapters must be injected before wrapping unless a backend-specific contract says otherwise.

## Checkpoint compatibility contract

`save_adapter_checkpoint` writes CPU safetensors plus `manifest.json`. The manifest requires `base_model_id`, `base_revision`, `architecture`, `configuration`, and `config_hash`. Adapter files are namespaced under `adapters/<name>/`; bridge/head/decoder files are separate. `load_adapter_checkpoint` requires matching base provenance and architecture. Canonical exports remain unmerged; use `merge_for_inference` only for a disposable inference copy or `compare_merged_unmerged` for a parity check.

## Architecture policy contract

- `vision`: attention projections and MLP projections, excluding patch/stem/convolution/norm/decoder paths.
- `3d_transformer`: only reviewed late Swin/transformer stages (`blocks.layers.1` in the tiny local fixture).
- `llm`: causal attention projections and MLP projections, excluding embeddings, norms, and heads.
- Unknown architectures require explicit target patterns and `confirm_target_modules=True`; zero matches and broad matches fail closed.

## Quantization matrix

- CUDA NF4 QLoRA is accepted only with supported bitsandbytes/Transformers runtime capability, CUDA placement, and a 4-bit-loaded model marker.
- TPU uses first-class BF16 LoRA/frozen-base preparation. bitsandbytes NF4 and unflagged experimental XLA quantization are rejected.
- `experimental_xla_quantization` remains explicitly labeled and is not part of acceptance.

## Phase 12 integration notes

The Phase 12 training engine should call the audit before constructing the optimizer, preserve `parameter_names` in run metadata, and save adapter-only checkpoints at stage boundaries. Task heads/decoders should be attached with `attach_trainable_module(..., role=...)` or included through `modules_to_save`; do not unfreeze a base module implicitly.
