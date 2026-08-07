# Phase 11 -> Phase 12 handoff

## Task module contract

Every task module consumes shared `EncoderOutput`/`MedicalBatch` semantics and returns `medfm.core.task.LossOutput`. Call `check_supported(batch.modality)` before the first batch. Keep task selection outside compiled forward or instantiate one fixed `MultiTaskLossComposer` signature per static bucket.

## Heads and decoders

- Pooled classification and box heads require `EncoderOutput.pooled_embedding` and fail with `UnsupportedCapabilityError` when absent.
- Spatial pooling requires `EncoderOutput.spatial_tokens` and honors `token_mask`; empty rows return finite zero representations.
- Segmentation decoders consume declared dense `feature_maps` and return `SegmentationOutput(logits, deep_supervision, native_outputs, auxiliary)`. Do not reshape spatial tokens into dense maps without declared feature-map semantics.
- `UNetDecoder2D/3D` and `FPNDecoder2D/3D` expect finest-resolution-last feature-map pyramids. `NativeModelDecoderWrapper` preserves opaque native results while requiring a spatial logits tensor.
- `LanguageConditionedMaskDecoder` takes visual maps plus separately encoded `[B,L,D]` text; query masks zero missing classes and output remains a spatial decoder result.

## Loss and reduction contract

- Use `DiceCELoss` for multiclass segmentation and `DiceBCELoss` for binary segmentation unless a task explicitly configures another loss.
- `LossOutput.components` names used by trainer logs: `classification`, `segmentation`, `language_segmentation`, `alignment`, `box_l1`, `box_iou`, and `language` for structured generation.
- `reduce_mean_by_count(local_mean, local_count, reduce_fn=...)` reduces `[loss * true_count, true_count]`; never average rank means. `reduce_loss_output` applies this to total and components and records `global_true_count`.
- Task diagnostics include detached `valid_count`, task names, selected weights, structured parse/schema counts, or spatial rank. Metric accumulation belongs outside compiled forward.

## PEFT integration

Attach task heads/decoders with `attach_trainable_module(..., role=...)` or include them through `modules_to_save`; do not implicitly unfreeze an encoder/base model. Run PEFT trainability audits before constructing the optimizer and preserve `parameter_names` in run metadata.

## Structured generation

Use `StructuredFindingsValidator`/`validate_generation_before_scoring` before any scorer. Invalid outputs stay in counts and diagnostics; raw payload retention requires `debug_access_controlled=True` and an explicit sink. Schema version is `STRUCTURED_FINDINGS_SCHEMA_VERSION` (currently 1).

## Deferred backend work

No TPU/XLA runtime or distributed process-group acceptance evidence is available locally. Phase 12 should run the fixed static task signature on the declared TPU buckets, use the reducer callback with all-reduce, and inspect XLA metrics for unsupported fallbacks before claiming backend parity.
