# Phase 18: CI, Hardening, and Release

## Objective

Prevent silent regressions, security failures, provenance loss, unsupported claims, and incompatible releases across the complete heterogeneous model stack.

## Dependencies

- [ ] All intended v1 phases and release-candidate recipes are accepted.
- [ ] Phase 17 produces candidate bundles and runtime documentation.
- [ ] Governance owners are available for final sign-off.

## Scope boundaries

Allowed areas: CI configuration, test markers/jobs, security checks, golden fixtures, release tooling, release docs, and final reports.

Do not weaken earlier acceptance gates to make the release pass.

## Implementation checklist

### Test matrix

- [ ] Level 1 CPU: schemas, registries, configs, losses, metrics, prompts, cache keys, and coordinates.
- [ ] Level 2 synthetic GPU: one 2D encoder, 3D encoder, pathology encoder, tiny QLoRA LM, decoder, and bridge.
- [ ] Level 3 protected real-checkpoint smoke: MedSigLIP, accepted 3D CT, accepted MRI/task model, accepted pathology model, and MedGemma.
- [ ] Level 4 golden regression: shapes, preprocess stats, logits, masks, structured fields, and memory envelopes.
- [ ] Keep weight/network tests explicitly marked and absent from ordinary CPU jobs.
- [ ] Fail on unexplained skipped tests or expired waivers.
- [ ] Add protected CUDA single-device, CUDA distributed, TPU replicated, and TPU SPMD/FSDP jobs according to the release support matrix.
- [ ] Require tiny/local models in routine accelerator jobs and schedule gated real-model jobs separately.
- [ ] Require every model registry entry to resolve CPU/CUDA/TPU status as supported, blocked, or explicitly untested.

### CI and quality gates

- [ ] Run formatting/linting, type checking, unit tests, coverage, package build, and docs/schema checks.
- [ ] Validate every phase report and completion manifest.
- [ ] Validate model/license/data registry consistency.
- [ ] Build/test containers and clean-environment bundle loading.
- [ ] Cache dependencies safely without caching credentials or patient data.
- [ ] Produce machine-readable test and coverage artifacts.
- [ ] Validate static TPU bucket manifests and XLA compile-count thresholds.
- [ ] Validate no TPU configuration enables bitsandbytes NF4.
- [ ] Validate no backend-neutral package imports CUDA/XLA-specific modules eagerly.

### Security and privacy

- [ ] Test manifest path traversal and unsafe URI handling.
- [ ] Test untrusted repository code, `trust_remote_code`, and malicious checkpoint rejection.
- [ ] Scan dependencies and container images for known vulnerabilities.
- [ ] Scan source/history/build artifacts for secrets and forbidden data patterns.
- [ ] Test PHI redaction in logs and exceptions.
- [ ] Test report prompt injection cannot trigger tool/system behavior.
- [ ] Test unauthorized download and license-policy bypass.
- [ ] Create a vulnerability reporting and incident response process.

### Reliability and regression

- [ ] Pin and document golden fixture generation.
- [ ] Set numeric tolerances by dtype/hardware rather than exact equality where necessary.
- [ ] Detect performance, peak-memory, output-schema, and preprocessing drift.
- [ ] Test interruption, checkpoint corruption, disk-full behavior, and cache corruption.
- [ ] Verify deterministic modes and record known nondeterministic kernels.
- [ ] Run a full install-to-inference rehearsal from a clean environment.
- [ ] Add cross-backend golden tolerances for outputs, gradients, updates, and metrics.
- [ ] Add portable adapter export/load regression across CPU, CUDA, and TPU.
- [ ] Track CUDA VRAM and TPU HBM/compile/steady-state performance separately.
- [ ] Detect new unsupported XLA operations or steady-state recompilation as regressions.

### Documentation and release artifacts

- [ ] Finalize README, architecture docs, user guides, operator runbooks, and model cards.
- [ ] Archive exact training/evaluation configs and provenance.
- [ ] Generate model/license/data summaries and checksums.
- [ ] List known limitations, blocked models, unsupported modalities/tasks, and hardware requirements.
- [ ] Remove unsupported clinical claims and clearly label research status.
- [ ] Define semantic versioning, migration, deprecation, and rollback policies.
- [ ] Produce signed/tagged release artifacts according to repository policy.
- [ ] Publish a model-by-task-by-backend support matrix with exact tested topology/runtime.
- [ ] Publish separate CUDA QLoRA and TPU BF16 LoRA guidance.
- [ ] State clearly that unsupported third-party custom CUDA models are not TPU-compatible.

### Final release gate

- [ ] All required tests pass.
- [ ] No unresolved critical/high security issue remains.
- [ ] No required acceptance result is `unknown`.
- [ ] Every included model has approved license and provenance.
- [ ] Every included dataset-derived result has documented provenance and split integrity.
- [ ] Validation report and representative errors are archived.
- [ ] Bundles load and pass smoke tests in the release environment.
- [ ] Product, engineering, clinical safety, data governance, and license owners sign off.
- [ ] Every claimed TPU workflow has a protected hardware report with compile and fallback counters.
- [ ] Every claimed distributed workflow has checkpoint/resume and valid-count metric evidence.

## Smoke command

```bash
make smoke
```

Protected accelerator smoke commands must additionally exercise the CUDA and TPU jobs declared by the release support matrix.

## Acceptance command

```bash
make lint && make typecheck && make test && python -m medfm.tools.validate_phase --phase 18
```

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [Primary implementation references](references.md)
- [PyTorch/XLA troubleshooting and metrics](https://docs.pytorch.org/xla/master/debug.html)
- [PyTorch/XLA profiling](https://docs.pytorch.org/xla/master/learn/xla-profiling.html)
- [PyTorch distributed checkpointing](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)

## Exit criteria

- [ ] CPU, synthetic GPU, protected real-checkpoint, and golden test levels pass as required.
- [ ] Security/privacy checks pass or have time-bound approved waivers.
- [ ] Release bundles reproduce accepted outputs in a clean environment.
- [ ] Documentation, provenance, licenses, and limitations are complete.
- [ ] The release contains no unsupported clinical claim.
- [ ] The release contains no unsupported accelerator or quantization claim.

## Handoff

- [ ] Publish release notes, checksums, compatibility matrix, and rollback steps.
- [ ] Archive all phase reports and final validation evidence.
- [ ] Open tracked follow-ups for deferred models/tasks and accepted waivers.
- [ ] Schedule periodic model-license, dependency-security, and regression reviews.
- [ ] Schedule periodic accelerator-runtime compatibility and TPU/GPU regression reviews.
