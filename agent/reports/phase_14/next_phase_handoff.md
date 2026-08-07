# Phase 14 handoff

## For Phase 15/16

- Consume `Phase14RecipeMetadata.to_dict()` as the run manifest source for
  modality, crop policy, shape buckets, token buckets, query/slice buckets,
  global-batch formula, memory cap, and backend observability.
- Keep native 3D and slice-sequence metrics/artifacts in separate namespaces;
  never compare a slice-sequence result as if it were a native-volume result.
- Use `native_3d_segmentation_metrics`, `native_vlm_grounding_metrics`, and
  `language_conditioned_segmentation_metrics` as the metric-family entry
  points. Preserve physical spacing for HD95 and volume error.
- Use `restore_volume_mask_to_original` only with recorded crop origin and
  transform history. Keep host inversion timing separate from accelerator
  timing in reports.
- The first TPU experiment should be the cached generic MONAI-compatible
  native-token profile. Run an operator audit and one-host PJRT acceptance
  before attempting joint native vision adaptation.
- Replace only the explicit offline-tiny adapter construction with approved
  local checkpoint loaders. Do not remove fail-closed production guards or
  silently substitute a different architecture.
- Before clinical claims, attach approved manifest hashes, external-site
  results, reader-review evidence, original-space visualizations, and actual
  CUDA/TPU memory/throughput measurements.

## Export candidates

- Frozen native 3D encoder plus classification head and segmentation decoder.
- Native 3D bridge/LLM adapters with cached-token metadata and coordinate
  provenance.
- Slice-sequence selector configuration plus fixed 2D tower/projector adapter.
- Language-conditioned 3D decoder with fixed query bucket.
