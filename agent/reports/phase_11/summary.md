# Phase 11 summary

Phase 11 delivers encoder-independent task heads, spatial decoders, task wrappers, baseline losses, structured-findings validation, and fixed-signature multitask composition without introducing a training loop or concrete-adapter dependency.

## Delivered

- Classification heads for pooled linear/MLP, attention pooling, multilabel, ordinal, and MIL inputs, with CLS, masked-mean, attention, GeM, top-k, and gated-MIL pooling operators. Spatial operators fail closed when only a pooled representation is available.
- Mandatory BCE-with-logits and cross-entropy baselines plus focal, label-smoothing, asymmetric multilabel, ordinal cumulative-link, Dice+CE, and Dice+BCE losses. Optional Tversky, boundary, focal-segmentation, class-volume, valid-voxel, and deep-supervision paths are explicit configuration.
- A shared `SegmentationOutput` contract preserving primary logits, deep-supervision outputs, native outputs, and auxiliary diagnostics. Static-shape-friendly 2D/3D UNet and FPN decoders, promptable/transformer mask interfaces, native decoder wrapping, and a 2D/3D language-conditioned spatial decoder are included.
- Task wrappers for classification, segmentation, language-conditioned segmentation, retrieval/alignment, localization, structured generation, and multitask composition. Every wrapper consumes shared `EncoderOutput`/batch contracts and reports named components, true valid counts, and diagnostics.
- Image/text projection with L2 normalization and learnable bounded logit scale, symmetric contrastive loss, same-patient negative filtering, and a distributed-negative provider protocol boundary.
- 2D/3D box heads, spatial box head, normalized/physical coordinate conversion with spacing and affine support, aligned IoU/GIoU, L1, and masked box losses.
- Versioned structured-findings JSON schema and validation-before-scoring helpers. Invalid output is counted with parse/schema diagnostics and is not retained unless an access-controlled debug sink is explicitly configured.
- Fixed and scheduled multitask weights, uncertainty/GradNorm extension protocol, fixed task signatures, finite/nonzero weight checks, and true-count distributed reduction helpers.
- Phase 11 schemas, synthetic fixtures, focused behavioral tests, smoke registration, and validator integration.

## Verification

- `python -m pytest tests/phase_11 -q`: 21 passed.
- `python -m medfm.tools.smoke --phase 11 --json`: passed.
- Phase 11 validator: passed after report artifacts were published.
- Ruff lint and formatting checks: passed for all changed Phase 11 modules, tools, and tests.
- CPU/CUDA mixed head, decoder, loss, projection, and backward smoke passed; a separate 3D CUDA segmentation forward/backward smoke also passed.

## Scope and backend notes

No generalized trainer, optimizer orchestration, concrete visual/language adapter import, CUDA custom extension, or raw-text persistence path was added. TPU/XLA execution and an actual distributed process-group run were not available on this workstation; policy-neutral tensor implementations and reduction tests remain in place without fabricating those results.
