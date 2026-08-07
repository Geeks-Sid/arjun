# Phase 14 3D recipe model cards

Phase 14 keeps native volumetric experiments distinct from slice-sequence
experiments. Every published recipe uses `recipe.phase: 14`, a pinned dataset
and preprocessing revision, and a fixed shape/token bucket. The offline tiny
profiles are deterministic contract fixtures only; they are not clinical
models.

## Native 3D classification

| Recipe | Encoder | Modality | Stage | Backend profile | Input policy |
| --- | --- | --- | --- | --- | --- |
| `classification_ct_fm_cuda.yaml` | CT-FM adapter | CT_3D | A frozen encoder + linear head | CUDA | fixed `[16,16,16]` contract crop |
| `classification_flexict3d.yaml` | FlexiCT-3D adapter | CT_3D | C final-stage vision LoRA | CPU contract; CUDA publication | full-volume policy with fixed tiny bucket |
| `classification_triad_mri.yaml` | Triad MAE adapter | MRI_3D | B attention pooling | CPU | low-resolution global crop |
| `classification_ct_fm_tpu_bf16.yaml` | generic MONAI-compatible adapter | CT_3D | A frozen encoder + linear head | TPU BF16 shape contract | one static `[16,16,16]` bucket |

Classification reports retain patient and study identifiers and expose
`per_patient` and `per_study` AUROC/AUPRC, calibration, and operating points.
The global batch formula is `microbatch_per_device * world_size *
gradient_accumulation_steps`.

## Native 3D segmentation

`segmentation_ct_fm_baseline.yaml` is the decoder-first baseline. It uses a
fixed native volume shape, foreground-lesion-centered sampling, a voxel
padding mask, and configurable Gaussian sliding-window blending. The MRI
`segmentation_triad_lora.yaml` profile adds final-stage vision LoRA and keeps
deep supervision explicitly opt-in. Native predictions can be re-embedded
using the recorded crop origin and spacing through
`restore_volume_mask_to_original`.

The metric contract includes Dice, surface Dice, sensitivity, HD95 in physical
units, lesion recall, false positives per scan, and absolute volume error in
mm³. Host transform and inversion timing are separate from accelerator
observability fields. `unsupported_xla_ops` and `custom_cuda_dependencies`
remain explicit in the metadata rather than being silently replaced.

## Native 3D VLM

`native_vlm_cached_tpu_bf16.yaml` is the first TPU bridge/LLM profile. It uses
32 visual tokens, fixed text length, a physical-coordinate encoder, a
Perceiver bridge, cached spatial tokens, and BF16 rather than bitsandbytes
quantization. `native_vlm_ct_fm_cuda_lora.yaml` is the CUDA joint-adaptation
profile with 64 visual tokens and late vision LoRA. Stages are:

1. bridge/alignment with frozen native vision and language components;
2. bridge plus language QLoRA/LoRA where the backend policy permits it;
3. late final-stage native vision LoRA;
4. optional region/box/mask/coordinate outputs after prior-stage evidence.

Image, no-image, and shuffled-image forward modes are separate observables.
The native VLM metadata always carries physical token coordinates and spacing;
a result from the slice family must not be reported as native 3D.

## Slice-sequence VLM

`slice_sequence_vlm_uniform.yaml` is a separate `MULTI_IMAGE_2D` experiment. A
host-side uniform selector chooses four slices from eight candidates and
preserves source index, normalized and physical z, series order, acquisition
window, MRI sequence, selector name, and selector revision. The 2D tower and
Perceiver operate on fixed per-slice shapes; selection occurs before collation.
Additional anatomy-aware, report-conditioned, entropy, lesion-aware, and
multi-window selectors are available through `build_slice_selector` and must
be evaluated as separate experiments. Slice count and visual/text token
budget rows are produced by `benchmark_slice_token_budgets` and are checked
against the configured 48 GB memory cap.

## Language-conditioned 3D segmentation

`language_conditioned_segmentation.yaml` uses explicit anatomy/lesion queries,
a fixed query bucket, text embeddings, cross-attention over the native 3D
feature pyramid, and a 3D mask decoder. The synthetic contract covers absent
targets, multiple targets, laterality, and ambiguous queries. Mask accuracy
and query grounding are reported as separate `per_query` metrics.

## Acceptance and limitations

The Phase 14 acceptance command is:

```bash
python -m pytest tests/phase_14 -q
python -m medfm.tools.validate_phase --phase 14
```

These profiles do not claim approved clinical weights, external-site
performance, human-reader review, or TPU hardware execution on every
backbone. Upstream checkpoint/license/access constraints remain fail-closed;
the generic MONAI-compatible path is the portable TPU contract baseline.
