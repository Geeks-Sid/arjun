# Phase 00: Requirements, Scope, and Governance

## Objective

Freeze the v1 product boundary, supported modalities and tasks, legal constraints, safety posture, and release acceptance criteria before implementation begins.

## Dependencies

- [ ] Confirm the workstation/GPU and PEFT-first assumptions in `idea.md`.
- [ ] Identify product, clinical safety, data governance, and model-license owners.

## Scope boundaries

Allowed areas: `docs/`, `model_registry/`, `agent/`, top-level policy files.

Do not implement runtime model, data, or training code in this phase.

## Implementation checklist

### Product and architecture

- [ ] Create `docs/product_requirements.md` with users, use cases, non-goals, and measurable v1 outcomes.
- [ ] Create `docs/supported_modalities.md` using the canonical names from `idea.md`.
- [ ] Create `docs/supported_tasks.md` using the canonical task names from `idea.md`.
- [ ] Map every modality/task pair to supported, deferred, or explicitly unsupported.
- [ ] Define the v1 model roster and identify a preferred and fallback backbone for each modality.
- [ ] State that billion-parameter full pretraining is outside v1.
- [ ] State that models remain modality-specific behind shared contracts.
- [ ] Define the minimum end-to-end vertical slices for 2D, 3D, WSI, segmentation, retrieval, and VLM work.
- [ ] Define CPU, single-GPU, multi-GPU, single-host TPU, and multi-host TPU support tiers.
- [ ] State that backend support is certified per model/task/topology, not framework-wide by assumption.
- [ ] Define the minimum v1 TPU target: tiny-model coverage for all task families plus at least one accepted real vision or language backbone.

### Governance and safety

- [ ] Create `docs/clinical_safety_scope.md` with intended-use exclusions and human-review requirements.
- [ ] Create `docs/data_governance.md` with de-identification, access, retention, audit, and deletion rules.
- [ ] Create `docs/model_governance.md` with model approval, review, deprecation, and incident processes.
- [ ] Create `docs/licensing_policy.md` separating research-only and commercially permitted artifacts.
- [ ] Create `docs/reproducibility_policy.md` listing mandatory run metadata and artifact retention.
- [ ] Define how PHI checks fail closed and where restricted report text may be stored.
- [ ] Define the policy for `trust_remote_code`, gated repositories, and untrusted checkpoints.
- [ ] Assign owners and review dates to every governance document.

### Architecture decisions

- [ ] Write ADR 0001: one framework with multiple backbones.
- [ ] Write ADR 0002: PEFT-first training.
- [ ] Write ADR 0003: external-encoder VLM bridge.
- [ ] Write ADR 0004: patient-level splitting.
- [ ] Write ADR 0005: native 3D and slice-sequence VLMs are distinct.
- [ ] Write ADR 0006: adapter-only checkpoints are canonical.
- [ ] Write ADR 0007: PyTorch with CUDA and PyTorch/XLA backends rather than separate GPU and TPU codebases.
- [ ] Write ADR 0008: static shape buckets and bounded token/tile/slice counts on TPU.
- [ ] Write ADR 0009: CUDA QLoRA versus TPU BF16 LoRA support policy.
- [ ] Record alternatives, consequences, and reversal conditions in every ADR.

### License registry seed

- [ ] Define the YAML license schema from `idea.md`.
- [ ] Add preliminary records for every v1 checkpoint.
- [ ] Record provider, source repository, revision policy, code license, and weights license.
- [ ] Record commercial use, redistribution, derivatives, gated access, approved use, and prohibited use.
- [ ] Mark unresolved terms as blocking rather than guessing.
- [ ] Require a named review owner and review date.

### Agent phase protocol

- [ ] Create `agent/README.md` and `agent/phase_template.md`.
- [ ] Create and validate `agent/acceptance_schema.json`.
- [ ] Create implement, review, test, and repair prompt templates under `agent/prompts/`.
- [ ] Define required phase report files and acceptance statuses.
- [ ] Require a tracked ADR for architectural changes discovered during implementation.

## Tests and verification

- [ ] Add a schema test for valid and invalid license records.
- [ ] Add a schema test for valid and invalid phase acceptance reports.
- [ ] Add a consistency test proving every supported modality has a backbone candidate.
- [ ] Add a consistency test proving every supported task has an implementation path.
- [ ] Add a link/reference check for internal policy documents.
- [ ] Add a policy consistency test requiring accelerator support status for every v1 model.
- [ ] Add a policy test rejecting a blanket cross-accelerator support claim without hardware evidence.

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

- [ ] Every requested modality and task has a disposition and implementation path.
- [ ] Every v1 model has a preliminary license record.
- [ ] Commercial and research-only usage is visibly separated.
- [ ] Clinical validation claims and exclusions are explicit.
- [ ] The agent phase protocol is complete enough for Phase 01 to use.

## Handoff

- [ ] List approved canonical enum values for Phase 02.
- [ ] List unresolved licenses and models that must remain disabled.
- [ ] List mandatory run metadata for Phase 01.
- [ ] List required accelerator tiers and v1 per-backend acceptance targets.
- [ ] Record policy owners and future review dates.
