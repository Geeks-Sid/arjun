# Phase 17: Inference, Export, and Serving

## Objective

Package adapter-based models into reproducible bundles and provide safe classification, segmentation, retrieval, VLM, and WSI inference without depending on training internals.

## Dependencies

- [ ] Accepted model checkpoints and evaluation artifacts exist.
- [ ] Phase 12 checkpoint schema and Phase 16 calibration/metric artifacts are stable.
- [ ] Phase 00 deployment licenses and safety policies approve each export.

## Scope boundaries

Allowed areas: `medfm/inference/`, export/bundle utilities, inference CLI, service layer, examples, and Phase 17 tests.

Do not expose unapproved research-only models in a production catalog or log raw images/reports by default.

## Implementation checklist

### Bundle format

- [ ] Implement the bundle layout from `idea.md` with schema versions.
- [ ] Include model card, license summary, base revisions, preprocess/postprocess, task schema, adapters, bridge, heads, calibration, examples, and checksums.
- [ ] Keep adapter-only artifacts canonical and optional merged artifacts secondary.
- [ ] Validate every file checksum and base-model compatibility before loading.
- [ ] Make bundle loading independent of the source training run directory.
- [ ] Include minimum runtime/dependency versions and hardware notes.
- [ ] Store canonical model/adapter/head/bridge tensors on CPU in accelerator-neutral safetensors where supported.
- [ ] Include tested CPU/CUDA/TPU support status and prohibited backend/loading combinations.
- [ ] Keep sharded resume checkpoints out of deployment bundles unless an explicit conversion step is documented.

### Inference pipelines

- [ ] Implement classification, segmentation, VLM, retrieval, and WSI commands.
- [ ] Validate request modality/task/schema before model allocation.
- [ ] Use the exact exported preprocessing and postprocessing config.
- [ ] Enforce batch, image, volume, tile, token, and output limits.
- [ ] Return structured errors without leaking sensitive input.
- [ ] Record deterministic input/output hashes where policy allows.
- [ ] Route through the same backend abstraction as training without hard-coded CUDA behavior.
- [ ] Predeclare fixed TPU inference buckets and reject or pad out-of-bucket inputs explicitly.
- [ ] Warm intended XLA buckets before latency measurement or serving readiness.

### 3D and DICOM interoperability

- [ ] Implement sliding-window inference with configurable overlap and Gaussian blending.
- [ ] Restore original orientation and spacing.
- [ ] Export NIfTI masks and verify reopen/coordinate integrity.
- [ ] Add DICOM SEG/SR/parametric output only through reviewed highdicom workflows.
- [ ] Preserve source references without exposing raw UIDs in general logs.
- [ ] Validate derived DICOM can be reopened and its geometry matches the source.

### VLM inference

- [ ] Default evaluation/clinical-style runs to deterministic decoding.
- [ ] Support greedy, justified beam search, and research sampling behind config.
- [ ] Implement JSON-constrained decoding where task schemas require it.
- [ ] Enforce max output, stop tokens, prompt version, and visual-token limits.
- [ ] Validate output schemas and expose uncertainty/invalid-output status.
- [ ] Prevent report text from altering system-level execution behavior.
- [ ] Keep TPU generation length buckets bounded and measure decode recompilation/host synchronization.

### Adapter serving and API

- [ ] Support one base LLM with multiple separately loaded adapters and bridges.
- [ ] Load only requested adapters and apply bounded cache/eviction behavior.
- [ ] Implement request validation, modality routing, preprocessing, loading, inference, postprocessing, and audit logging.
- [ ] Add concurrency/backpressure and timeout controls.
- [ ] Avoid dynamic unreviewed model code and arbitrary file paths.
- [ ] Version API request and response schemas.
- [ ] Keep CUDA and TPU worker pools/configuration separate while preserving common request/result schemas.
- [ ] Test adapter switching does not retain stale device/sharded state.

### Auditing and privacy

- [ ] Record model/adapter revision, preprocess hash, prompt version, input hash, time, schema, runtime, VRAM, and error status.
- [ ] Redact reports/images/identifiers by default.
- [ ] Separate operational logs from access-controlled clinical audit data.
- [ ] Test deletion/retention policy and log access controls in deployment documentation.

## Tests and verification

- [ ] Load an adapter-only bundle in a clean environment.
- [ ] Verify wrong/missing base revisions and checksum mismatch fail closed.
- [ ] Compare exported and training-repository outputs within tolerance.
- [ ] Restore 3D masks to original coordinates.
- [ ] Reopen NIfTI and DICOM-derived output.
- [ ] Validate VLM structured output and deterministic repeatability.
- [ ] Test request limits, timeout, corrupt input, and concurrent adapter switching.
- [ ] Verify audit logs contain required fields and no raw sensitive payload.
- [ ] Assert configured inference memory caps on representative workloads.
- [ ] Load one portable adapter bundle on CPU, CUDA, and TPU and compare outputs within tolerance.
- [ ] Record CUDA warm latency separately from TPU compile warmup and steady-state latency.
- [ ] Assert stable XLA compilation counts for repeated fixed-bucket inference.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [MONAI Bundles](https://docs.monai.io/en/stable/bundle_intro.html)
- [highdicom](https://highdicom.readthedocs.io/en/latest/)
- [PyTorch/XLA tensor save/load guidance](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html)

## Smoke command

```bash
python -m medfm.cli.infer classification --config configs/smoke/inference.yaml
```

## Acceptance command

```bash
pytest tests/phase_17 -q && python -m medfm.tools.validate_phase --phase 17
```

## Exit criteria

- [ ] Adapter-only exports load cleanly and reproduce expected outputs.
- [ ] 3D and DICOM outputs preserve geometry and reopen successfully.
- [ ] VLM outputs pass explicit schema validation.
- [ ] Service limits, auditing, privacy, and adapter switching pass tests.
- [ ] Representative workloads stay below the configured memory cap.
- [ ] At least one accelerator-neutral adapter bundle passes CPU/CUDA/TPU load and parity tests.

## Handoff

- [ ] Publish bundle/API schemas and supported runtime matrix.
- [ ] Publish deployment license catalog and blocked bundles.
- [ ] Publish performance/memory/latency measurements.
- [ ] Provide release artifacts and operational runbooks to Phase 18.
- [ ] Publish backend-specific warmup, bucket, capacity, and latency runbooks.
