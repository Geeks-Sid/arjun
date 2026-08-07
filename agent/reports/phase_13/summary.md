# Phase 13 summary

Phase 13 delivers reproducible 2D classification, segmentation, native VLM, and external-encoder VLM recipe contracts on top of the Phase 12 training engine. Recipe-owned builders keep model and dataset choices out of the trainer, while every offline contract records revisions, fixed shapes, token buckets, task targets, and staged trainable roles.

## Delivered

- Added a typed Phase 13 recipe builder covering classification, segmentation, promptable segmentation, native MedGemma-style VLM, and external bridge-to-LLM VLM families.
- Added a classification matrix for MedSigLIP, RAD-DINO, H-Optimus tile, FlexiCT-2D, frozen smoke, multilabel smoke, CUDA BF16, TPU BF16, and staged visual-LoRA development.
- Added frozen decoder and promptable segmentation paths with prompt tensors kept separate from image pixels, original-coordinate mask restoration, static crop/tile settings, and per-class clinical-unit metrics.
- Added native VLM VQA and structured-findings smoke paths with prompt masking, no-cache training, fixed visual-token buckets, explicit mixed-task weights, and schema validation through `StructuredGenerationTask`.
- Added external linear and Perceiver bridge configurations for 32/64/128 visual-token buckets, cached TPU baseline metadata, CUDA QLoRA policy configuration, and staged bridge/language/vision role declarations.
- Added deterministic classification and segmentation metric interfaces, reproducibility evaluation artifacts, and image/no-image/shuffled-image visual-dependence ablation reporting.
- Fixed optimizer role aliases for language/task/decoder/LoRA parameters and included optimizer components in trainable/total run-metadata accounting. Metadata now records the configured gradient accumulation rather than defaulting to one.
- Added Phase 13 focused tests and registered the Phase 13 validator and required artifacts.
- Published `docs/recipes/phase13_2d_model_cards.md` with family-specific provenance, safety limits, and production evidence gates.

## Verification

- `python -m pytest tests/phase_13 -q`: 19 passed.
- Final CPU recipe smokes passed:
  - classification: one optimizer step, effective batch 2, 33 trainable parameters.
  - segmentation: two optimizer steps, effective batch 2, 1,498 trainable parameters.
  - native VLM: one optimizer step, effective batch 2, 81,472 trainable parameters.
  - external VLM: one optimizer step, effective batch 2, 8,448 trainable parameters.
- Multilabel classification, promptable segmentation, structured findings, and metadata accumulation checks passed.
- Compatible classification checkpoint resume passed at global step 1; changing `max_steps` was correctly rejected by the config-hash guard.
- `python -m medfm.tools.validate_phase --phase 13`: passed after the report artifacts were populated.

## Runtime limits

This workstation has no `torch_xla`, `bitsandbytes`, or approved production checkpoints. TPU PJRT/BF16 execution, CUDA NF4 QLoRA execution, multi-device distributed recipe acceptance, representative large-model VRAM, protected clinical datasets, external-site grounding, and human-review evidence were not executed and are not presented as passing results. The published profiles fail closed or remain explicit production handoff configurations.
