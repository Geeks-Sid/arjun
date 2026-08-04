# Phase 00: Requirements, Scope, and Governance

## Objective

Freeze the v1 product boundary, supported modalities and tasks, legal constraints, safety posture, and release acceptance criteria before implementation begins.

## Dependencies

- [x] Confirm the workstation/GPU and PEFT-first assumptions in `idea.md`.
- [x] Identify product, clinical safety, data governance, and model-license owners.

## Scope boundaries

Allowed areas: `docs/`, `model_registry/`, `agent/`, top-level policy files.

Do not implement runtime model, data, or training code in this phase.

## Implementation checklist

### Product and architecture

- [x] Create `docs/product_requirements.md` with users, use cases, non-goals, and measurable v1 outcomes.
- [x] Create `docs/supported_modalities.md` using the canonical names from `idea.md`.
- [x] Create `docs/supported_tasks.md` using the canonical task names from `idea.md`.
- [x] Map every modality/task pair to supported, deferred, or explicitly unsupported.
- [x] Define the v1 model roster and identify a preferred and fallback backbone for each modality.
- [x] State that billion-parameter full pretraining is outside v1.
- [x] State that models remain modality-specific behind shared contracts.
- [x] Define the minimum end-to-end vertical slices for 2D, 3D, WSI, segmentation, retrieval, and VLM work.
- [x] Define CPU, single-GPU, multi-GPU, single-host TPU, and multi-host TPU support tiers.
- [x] State that backend support is certified per model/task/topology, not framework-wide by assumption.
- [x] Define the minimum v1 TPU target: tiny-model coverage for all task families plus at least one accepted real vision or language backbone.

### Governance and safety

- [x] Create `docs/clinical_safety_scope.md` with intended-use exclusions and human-review requirements.
- [x] Create `docs/data_governance.md` with de-identification, access, retention, audit, and deletion rules.
- [x] Create `docs/model_governance.md` with model approval, review, deprecation, and incident processes.
- [x] Create `docs/licensing_policy.md` separating research-only and commercially permitted artifacts.
- [x] Create `docs/reproducibility_policy.md` listing mandatory run metadata and artifact retention.
- [x] Define how PHI checks fail closed and where restricted report text may be stored.
- [x] Define the policy for `trust_remote_code`, gated repositories, and untrusted checkpoints.
- [x] Assign owners and review dates to every governance document.

### Architecture decisions

- [x] Write ADR 0001: one framework with multiple backbones.
- [x] Write ADR 0002: PEFT-first training.
- [x] Write ADR 0003: external-encoder VLM bridge.
- [x] Write ADR 0004: patient-level splitting.
- [x] Write ADR 0005: native 3D and slice-sequence VLMs are distinct.
- [x] Write ADR 0006: adapter-only checkpoints are canonical.
- [x] Write ADR 0007: PyTorch with CUDA and PyTorch/XLA backends rather than separate GPU and TPU codebases.
- [x] Write ADR 0008: static shape buckets and bounded token/tile/slice counts on TPU.
- [x] Write ADR 0009: CUDA QLoRA versus TPU BF16 LoRA support policy.
- [x] Record alternatives, consequences, and reversal conditions in every ADR.

### License registry seed

- [x] Define the YAML license schema from `idea.md`.
- [x] Add preliminary records for every v1 checkpoint.
- [x] Record provider, source repository, revision policy, code license, and weights license.
- [x] Record commercial use, redistribution, derivatives, gated access, approved use, and prohibited use.
- [x] Mark unresolved terms as blocking rather than guessing.
- [x] Require a named review owner and review date.

### Agent phase protocol

- [x] Create `agent/README.md` and `agent/phase_template.md`.
- [x] Create and validate `agent/acceptance_schema.json`.
- [x] Create implement, review, test, and repair prompt templates under `agent/prompts/`.
- [x] Define required phase report files and acceptance statuses.
- [x] Require a tracked ADR for architectural changes discovered during implementation.

## Tests and verification

- [x] Add a schema test for valid and invalid license records.
- [x] Add a schema test for valid and invalid phase acceptance reports.
- [x] Add a consistency test proving every supported modality has a backbone candidate.
- [x] Add a consistency test proving every supported task has an implementation path.
- [x] Add a link/reference check for internal policy documents.
- [x] Add a policy consistency test requiring accelerator support status for every v1 model.
- [x] Add a policy test rejecting a blanket cross-accelerator support claim without hardware evidence.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [Primary implementation references](references.md)
- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/master/)
- [Transformers bitsandbytes hardware compatibility](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

## Smoke command

```bash
python -m pytest tests/phase_00 -q
```

## Acceptance command

```bash
python -m medfm.tools.validate_phase --phase 00
```

## Exit criteria

- [x] Every requested modality and task has a disposition and implementation path.
- [x] Every v1 model has a preliminary license record.
- [x] Commercial and research-only usage is visibly separated.
- [x] Clinical validation claims and exclusions are explicit.
- [x] The agent phase protocol is complete enough for Phase 01 to use.

## Handoff

- [x] List approved canonical enum values for Phase 02.
- [x] List unresolved licenses and models that must remain disabled.
- [x] List mandatory run metadata for Phase 01.
- [x] List required accelerator tiers and v1 per-backend acceptance targets.
- [x] Record policy owners and future review dates.
