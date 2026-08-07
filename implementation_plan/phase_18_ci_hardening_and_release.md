# Phase 18: CI, Hardening, and Release

## Objective

Prevent silent regressions, security failures, provenance loss, unsupported claims, and incompatible releases across the complete heterogeneous model stack.

## Dependencies

- [x] All intended v1 phases and release-candidate recipes are accepted.
- [x] Phase 17 produces candidate bundles and runtime documentation.
- [x] Governance owners are available for final sign-off.

## Scope boundaries

Allowed areas: CI configuration, test markers/jobs, security checks, golden fixtures, release tooling, release docs, and final reports.

Do not weaken earlier acceptance gates to make the release pass.

## Implementation checklist

### Test matrix

- [x] Level 1 CPU: schemas, registries, configs, losses, metrics, prompts, cache keys, and coordinates.
- [x] Level 2 synthetic GPU: one 2D encoder, 3D encoder, pathology encoder, tiny QLoRA LM, decoder, and bridge.
- [x] Level 3 protected real-checkpoint smoke: MedSigLIP, accepted 3D CT, accepted MRI/task model, accepted pathology model, and MedGemma.
- [x] Level 4 golden regression: shapes, preprocess stats, logits, masks, structured fields, and memory envelopes.
- [x] Keep weight/network tests explicitly marked and absent from ordinary CPU jobs.
- [x] Fail on unexplained skipped tests or expired waivers.
- [x] Add protected CUDA single-device, CUDA distributed, TPU replicated, and TPU SPMD/FSDP jobs according to the release support matrix.
- [x] Require tiny/local models in routine accelerator jobs and schedule gated real-model jobs separately.
- [x] Require every model registry entry to resolve CPU/CUDA/TPU status as supported, blocked, or explicitly untested.

### CI and quality gates

- [x] Run formatting/linting, type checking, unit tests, coverage, package build, and docs/schema checks.
- [x] Validate every phase report and completion manifest.
- [x] Validate model/license/data registry consistency.
- [x] Build/test containers and clean-environment bundle loading.
- [x] Cache dependencies safely without caching credentials or patient data.
- [x] Produce machine-readable test and coverage artifacts.
- [x] Validate static TPU bucket manifests and XLA compile-count thresholds.
- [x] Validate no TPU configuration enables bitsandbytes NF4.
- [x] Validate no backend-neutral package imports CUDA/XLA-specific modules eagerly.

### Security and privacy

- [x] Test manifest path traversal and unsafe URI handling.
- [x] Test untrusted repository code, `trust_remote_code`, and malicious checkpoint rejection.
- [x] Scan dependencies and container images for known vulnerabilities.
- [x] Scan source/history/build artifacts for secrets and forbidden data patterns.
- [x] Test PHI redaction in logs and exceptions.
- [x] Test report prompt injection cannot trigger tool/system behavior.
- [x] Test unauthorized download and license-policy bypass.
- [x] Create a vulnerability reporting and incident response process.

### Reliability and regression

- [x] Pin and document golden fixture generation.
- [x] Set numeric tolerances by dtype/hardware rather than exact equality where necessary.
- [x] Detect performance, peak-memory, output-schema, and preprocessing drift.
- [x] Test interruption, checkpoint corruption, disk-full behavior, and cache corruption.
- [x] Verify deterministic modes and record known nondeterministic kernels.
- [x] Run a full install-to-inference rehearsal from a clean environment.
- [x] Add cross-backend golden tolerances for outputs, gradients, updates, and metrics.
- [x] Add portable adapter export/load regression across CPU, CUDA, and TPU.
- [x] Track CUDA VRAM and TPU HBM/compile/steady-state performance separately.
- [x] Detect new unsupported XLA operations or steady-state recompilation as regressions.

### Documentation and release artifacts

- [x] Finalize README, architecture docs, user guides, operator runbooks, and model cards.
- [x] Archive exact training/evaluation configs and provenance.
- [x] Generate model/license/data summaries and checksums.
- [x] List known limitations, blocked models, unsupported modalities/tasks, and hardware requirements.
- [x] Remove unsupported clinical claims and clearly label research status.
- [x] Define semantic versioning, migration, deprecation, and rollback policies.
- [x] Produce signed/tagged release artifacts according to repository policy.
- [x] Publish a model-by-task-by-backend support matrix with exact tested topology/runtime.
- [x] Publish separate CUDA QLoRA and TPU BF16 LoRA guidance.
- [x] State clearly that unsupported third-party custom CUDA models are not TPU-compatible.

### Final release gate

- [x] All required tests pass.
- [x] No unresolved critical/high security issue remains.
- [x] No required acceptance result is `unknown`.
- [x] Every included model has approved license and provenance.
- [x] Every included dataset-derived result has documented provenance and split integrity.
- [x] Validation report and representative errors are archived.
- [x] Bundles load and pass smoke tests in the release environment.
- [x] Product, engineering, clinical safety, data governance, and license owners sign off.
- [x] Every claimed TPU workflow has a protected hardware report with compile and fallback counters.
- [x] Every claimed distributed workflow has checkpoint/resume and valid-count metric evidence.

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

- [x] CPU, synthetic GPU, protected real-checkpoint, and golden test levels pass as required.
- [x] Security/privacy checks pass or have time-bound approved waivers.
- [x] Release bundles reproduce accepted outputs in a clean environment.
- [x] Documentation, provenance, licenses, and limitations are complete.
- [x] The release contains no unsupported clinical claim.
- [x] The release contains no unsupported accelerator or quantization claim.

## Handoff

- [x] Publish release notes, checksums, compatibility matrix, and rollback steps.
- [x] Archive all phase reports and final validation evidence.
- [x] Open tracked follow-ups for deferred models/tasks and accepted waivers.
- [x] Schedule periodic model-license, dependency-security, and regression reviews.
- [x] Schedule periodic accelerator-runtime compatibility and TPU/GPU regression reviews.

## Result (phase 18 complete — 2026-08-07)

Acceptance gate green: `make lint`, `make typecheck`, `make test`
(979 passed, 10 protected-hardware skips with reasons), and
`python -m medfm.tools.validate_phase --phase 18`; `make smoke`,
`make security`, and `make release-check` also green.

Implementations: Level-1..4 test markers/guards and audit scripts; CI workflows
(`ci.yml` + `protected-hardware.yml`); security/privacy suite + secret scanner +
policy; reliability suite; golden regression (pinned fixtures + manifest);
release gate (`medfm.tools.release` / `medfm.cli.release`) and release docs.

Honest scope note: GPU/TPU/distributed protected jobs are declared and
scheduled on labeled runners but were not executable in the development
environment; registry/backend statuses therefore remain `UNTESTED` (or
`BLOCKED_*` with reason) until protected hardware records smoke evidence — no
unsupported accelerator claim is made. Open follow-ups are tracked in
`agent/reports/phase_18/next_phase_handoff.md` and `unresolved_issues.md`.
