# Phase 00 Summary — Requirements, Scope, and Governance

Date: 2026-08-04
Agent: Kimi Code CLI

## What was done

- Froze the v1 product boundary in `docs/product_requirements.md`: users, use cases, non-goals (billion-parameter full pretraining excluded; modality-specific models behind shared contracts), measurable outcomes.
- Confirmed platform assumptions: the single 48 GB CUDA GPU is the v1 build target (project owner direction, 2026-08-04); the local dev shell's 8 GB RTX 4060 Laptop GPU is a development convenience only. PEFT-first confirmed.
- Defined canonical modality (10) and task (16) enums, and a complete modality × task disposition matrix (supported/deferred/unsupported) in `model_registry/v1_scope.yaml`, summarized in `docs/supported_modalities.md` and `docs/supported_tasks.md`.
- Defined the v1 model roster (16 v1 models + 1 deferred) with preferred/fallback backbones per modality and six minimum vertical slices (2D, 3D, WSI, segmentation, retrieval, VLM).
- Defined accelerator tiers T0–T5, per-model/task/topology certification, and the minimum v1 TPU target (tiny-model coverage of all task families + at least one real HF vision or language backbone).
- Wrote the five governance documents (clinical safety, data, model, licensing, reproducibility), each with an owner and review date (2026-11-02). PHI checks fail closed; restricted report text storage rules defined; `trust_remote_code`/gated-repo/untrusted-checkpoint policy defined.
- Wrote ADRs 0001–0009, each with alternatives, consequences, and reversal conditions.
- Seeded the license registry: `model_registry/license_schema.json` (YAML structure from `idea.md`) plus preliminary records for every roster model. Unresolved terms are blocking (`blocked_unresolved`), never guessed.
- Created the agent phase protocol (`agent/`): README, phase template, acceptance schema, implement/review/test/repair prompts, report structure, acceptance statuses (`passed`/`failed`/`blocked`/`not_applicable`; `unknown` illegal).
- Implemented `medfm.tools.validate_phase` and `medfm.tools.governance`, and the `tests/phase_00/` suite (schema, consistency, link, and accelerator-policy tests).

## Key decisions

- 48 GB single GPU is the planning baseline even though the dev shell is smaller (owner direction).
- Most roster models start life disabled: only models with reviewed, permissive licenses can move to `approved_*`; everything unresolved is `blocked_unresolved`.
- Backend support is per model/task/topology with mandatory evidence; blanket cross-accelerator claims are rejected by test.
