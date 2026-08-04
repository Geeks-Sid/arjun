# Phase 05: Model Registry, Capabilities, and Weight Management

## Objective

Create a plugin registry that declares each model's exact capabilities, preprocessing, license status, revisions, loading modes, memory profile, and safe weight acquisition behavior.

## Dependencies

- [ ] Phases 00-04 are accepted.
- [ ] License schema and approval policy are available.
- [ ] Core input/output/preprocess contracts are stable.

## Scope boundaries

Allowed areas: `medfm/registry/`, core registry glue, `model_registry/`, model CLI, weight cache utilities, and Phase 05 tests.

Do not implement model architecture adapters beyond tiny/dummy plugins.

## Implementation checklist

### Registry schema and lifecycle

- [ ] Implement `ModelSpec`, `LicenseSpec`, `MemoryProfile`, and PEFT capability schemas.
- [ ] Require modality/task support, input/output specs, feature capabilities, repository, and pinned revision.
- [ ] Distinguish native text support, pooled embeddings, spatial tokens, feature maps, and hidden states.
- [ ] Record supported loading modes and architecture-specific LoRA target policies.
- [ ] Version registry records and support deprecation/replacement metadata.
- [ ] Reject duplicate model IDs and ambiguous aliases.
- [ ] Add per-backend support status for CPU, CUDA single/multi-device, and TPU replicated/sharded modes.
- [ ] Record tested precision, quantization, attention backend, custom operators, topology, and last successful smoke revision.

### License and governance gates

- [ ] Require every production-loadable model to have a complete license record.
- [ ] Block unknown, expired, or unapproved license status.
- [ ] Separate research and deployment catalogs.
- [ ] Store gated-access acceptance outside source-controlled model records.
- [ ] Never automatically accept terms or imply acceptance from a successful download.
- [ ] Include approved and prohibited use cases in model inspection output.

### Weight acquisition and integrity

- [ ] Implement explicit download, inspect, and local-path resolution operations.
- [ ] Pin repository revisions/commit SHAs.
- [ ] Prefer safetensors and reject unsafe formats by policy unless reviewed.
- [ ] Verify file hashes and expected file sets.
- [ ] Use atomic downloads and resume safely.
- [ ] Keep access tokens out of commands, logs, and manifests.
- [ ] Default `trust_remote_code` to false and require a reviewed allowlist exception.

### Capability discovery

- [ ] Implement registry queries by modality, task, loading mode, and license class.
- [ ] Validate preprocessing requirements are complete.
- [ ] Validate output capabilities agree with task declarations.
- [ ] Record whether spatial tokens are native, hooked, or unavailable.
- [ ] Record known LoRA target modules and unknown-family confirmation requirements.
- [ ] Add a blocked status with a structured reason for unavailable checkpoints.
- [ ] Detect CUDA-only extensions and require a tested pure-PyTorch/SDPA path before TPU eligibility.
- [ ] Distinguish `UNTESTED` from `SUPPORTED`; source inspection alone cannot promote accelerator support.

### CLI

- [ ] Implement `medfm models list` with filters.
- [ ] Implement `show`, `validate`, `download`, `smoke`, `inspect-modules`, and `estimate-memory`.
- [ ] Make JSON output available for automation.
- [ ] Ensure commands distinguish metadata validation, local weight validation, and runtime smoke status.
- [ ] Keep network access opt-in and explicit.
- [ ] Add `--backend` filters and an accelerator compatibility report.
- [ ] Implement `medfm accelerator validate-model` using the exact model revision and tiny backend-specific input.

### Loading modes and memory estimates

- [ ] Represent full, BF16, FP16, INT8, NF4, CPU offload, and frozen-cache modes.
- [ ] Define conservative estimate inputs and uncertainty notes.
- [ ] Reject unsupported model/mode combinations before model construction.
- [ ] Record actual peak memory later to refine estimates.
- [ ] Add TPU BF16 replicated and TPU BF16 SPMD/FSDP loading modes.
- [ ] Reject bitsandbytes NF4 when `xla_tpu` is selected.
- [ ] Record compile risk from dynamic shapes/custom operators separately from memory estimates.

## Tests and verification

- [ ] Reject incomplete license records.
- [ ] Reject missing normalization, unpinned revision, and contradictory capabilities.
- [ ] Reject duplicate IDs and unsafe aliases.
- [ ] Verify research-only models never appear in the deployment catalog.
- [ ] Verify hash mismatch and partial download handling.
- [ ] Verify no network is used by list/show/validate metadata commands.
- [ ] Verify a dummy model plugin completes a local smoke test.
- [ ] Verify every run artifact receives exact model ID and revision.
- [ ] Verify CUDA success does not mutate TPU status from `UNTESTED`.
- [ ] Verify unsupported quantization/backend combinations fail before weight allocation.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [Primary model and framework sources](references.md)
- [Transformers bitsandbytes hardware compatibility](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [PyTorch/XLA troubleshooting and unsupported-op counters](https://docs.pytorch.org/xla/master/debug.html)

## Smoke command

```bash
python -m medfm.cli.models list --format json
```

## Acceptance command

```bash
pytest tests/phase_05 -q && python -m medfm.tools.validate_phase --phase 05
```

## Exit criteria

- [ ] Invalid licenses and preprocess specs fail closed.
- [ ] The CLI exposes capability and provenance data reliably.
- [ ] Weight downloads are explicit, pinned, integrity-checked, and credential-safe.
- [ ] Real v1 models are registered as ready or blocked with reasons.
- [ ] Phases 06-08 can add adapters without changing registry internals.

## Handoff

- [ ] Publish adapter registration instructions and dummy plugin example.
- [ ] Publish license and model smoke checklists.
- [ ] List blocked v1 checkpoints and owners.
- [ ] Freeze registry schema version for adapter phases.
- [ ] Publish the accelerator capability status vocabulary and evidence requirements.
