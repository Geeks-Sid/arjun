# Phase 11: Task Heads, Decoders, and Losses

## Objective

Implement encoder-independent modules for classification, segmentation, retrieval, localization, structured generation, language conditioning, and multitask loss composition.

## Dependencies

- [ ] Phase 02 task/encoder output contracts are accepted.
- [ ] Representative 2D, 3D, and WSI dummy outputs exist.
- [ ] Phase 09 defines language and visual token output semantics.

## Scope boundaries

Allowed areas: `medfm/models/heads/`, `models/decoders/`, `medfm/tasks/`, task schemas, and Phase 11 tests.

Do not depend on concrete encoder classes or implement training-loop orchestration.

## Implementation checklist

### Classification

- [ ] Implement linear, MLP, attention-pooling, multilabel, ordinal, and MIL heads.
- [ ] Implement CLS, masked mean, attention, generalized mean, top-k, and MIL pooling.
- [ ] Validate pooled/spatial capability requirements before forward.
- [ ] Implement BCE-with-logits, cross-entropy, class weights, focal, label smoothing, asymmetric multilabel, and ordinal losses.
- [ ] Establish plain BCE/cross-entropy as mandatory baselines.

### Segmentation

- [ ] Define a shared decoder output contract for logits, deep supervision, and native outputs.
- [ ] Implement minimal UNet decoders for 2D and 3D.
- [ ] Implement FPN variants where feature pyramids are available.
- [ ] Add native model decoder wrappers without flattening native semantics.
- [ ] Add promptable and transformer-mask decoder interfaces.
- [ ] Implement Dice + CE and Dice + BCE defaults.
- [ ] Add focal, Tversky, boundary, deep-supervision, and class-volume options behind explicit config.

### Language-conditioned segmentation

- [ ] Encode a text query separately from mask generation.
- [ ] Implement cross-attention between text embeddings and visual feature maps.
- [ ] Produce masks through a spatial decoder, never raw text tokens.
- [ ] Support 2D and 3D feature shapes through one task interface.
- [ ] Verify query masking and missing-class behavior.

### Retrieval and contrastive alignment

- [ ] Implement image/text projections, L2 normalization, and learnable logit scale.
- [ ] Implement symmetric contrastive loss.
- [ ] Filter same-patient negatives when configured.
- [ ] Define an interface for future distributed negatives.
- [ ] Prevent invalid all-positive/all-filtered batches.

### Localization and structured generation

- [ ] Implement 2D/3D box heads and normalized/physical coordinate conversion.
- [ ] Implement L1 and IoU/GIoU-style losses.
- [ ] Define versioned structured findings JSON schema from `idea.md`.
- [ ] Validate generation before scoring and report parse/schema errors.
- [ ] Preserve invalid raw output only in access-controlled debug artifacts.

### Multitask composition

- [ ] Implement named classification, segmentation, language, alignment, and box losses.
- [ ] Support fixed and scheduled weights first.
- [ ] Define extension points for uncertainty and GradNorm weighting.
- [ ] Validate active tasks have nonzero, finite weights and compatible outputs.
- [ ] Return per-task counts and diagnostics.
- [ ] Keep task selection outside compiled forward or use a fixed task signature per static bucket.
- [ ] Reduce each distributed loss by true sample/token/voxel counts, including padded-batch masks.

### Accelerator-safe implementation

- [ ] Use ordinary PyTorch tensor operations with CPU, CUDA, and XLA lowering.
- [ ] Avoid CUDA custom extensions in baseline losses/heads.
- [ ] Avoid `.item()`, tensor-dependent Python branching, and device-to-host synchronization inside the training step.
- [ ] Keep metric accumulation detached and outside gradient graphs.
- [ ] Keep numerically sensitive reductions in FP32 on CUDA and TPU.
- [ ] Define static output shapes for each task/bucket, including empty-target cases.

## Tests and verification

- [ ] Feed every head a generic `EncoderOutput` fixture.
- [ ] Verify no task imports a concrete adapter class.
- [ ] Test loss values against hand-computed small examples.
- [ ] Test 2D and 3D segmentation shapes, empty masks, and all-positive masks.
- [ ] Verify language-conditioned mask gradients reach text/visual fusion.
- [ ] Verify contrastive same-patient filtering.
- [ ] Verify physical box conversion with nontrivial affine/spacing.
- [ ] Verify invalid structured output is counted, not silently dropped.
- [ ] Combine classification, segmentation, and language loss in one synthetic step.
- [ ] Run head/loss forward/backward parity tests on CPU, CUDA, and TPU.
- [ ] Test distributed reductions with uneven valid counts and padded entries.
- [ ] Inspect XLA metrics for unsupported-op fallbacks in every baseline task family.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [PyTorch/XLA migration guide](https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html)
- [MONAI losses](https://docs.monai.io/en/stable/losses.html)

## Smoke command

```bash
python -m medfm.tools.smoke --phase 11
```

## Acceptance command

```bash
pytest tests/phase_11 -q && python -m medfm.tools.validate_phase --phase 11
```

## Exit criteria

- [ ] All task modules consume shared contracts only.
- [ ] 2D and 3D segmentation share a task interface.
- [ ] Baseline losses pass numerical tests.
- [ ] Multitask losses coexist in one backward pass.
- [ ] Invalid structured output remains visible in metrics and reports.
- [ ] Baseline heads/losses pass CUDA and TPU parity within declared tolerances.

## Handoff

- [ ] Publish task configuration schemas and required capabilities.
- [ ] Publish loss/metric names used by trainer logs.
- [ ] Provide one-batch synthetic fixtures to Phase 12.
- [ ] Record optional advanced heads/losses deferred beyond baseline acceptance.
- [ ] Publish valid-count reduction semantics and backend parity tolerances.
