# ADR 0001: One framework with multiple backbones

Status: Accepted (2026-08-04)
Deciders: Project Maintainer

## Context

The project must support 2D radiology, 3D CT/MRI, WSI pathology, retrieval, segmentation, and VLM tasks. Each modality has strong pretrained backbones (MedSigLIP, RAD-DINO, CT-FM, Triad, H-Optimus-0, MedGemma) with heterogeneous interfaces.

## Decision

Build **one framework** with shared contracts (sample/batch schemas, `VisualEncoder`, `LanguageModelAdapter`, `TaskModule` protocols, shared PEFT engine, trainer, and checkpoint format) and per-modality adapter implementations. Models remain modality-specific internally; unification happens only at the contract layer.

## Alternatives considered

- **Separate per-modality codebases:** faster per modality, but duplicated training/eval/checkpoint logic and no shared governance gates. Rejected.
- **One monolithic multimodal model:** outside v1 scope and contradicted by the modality-specific nature of the backbones. Rejected (also a product non-goal).
- **Adopt an existing framework wholesale (MONAI bundles / HF only):** neither covers WSI + 3D + VLM + governance uniformly; we use MONAI and HF as libraries, not as the framework. Rejected.

## Consequences

- Every new backbone is an adapter behind `VisualEncoder`; new tasks are `TaskModule`s. Onboarding cost per model is bounded.
- Contract tests (Phase 02) become the enforcement point and are mandatory.
- Shared governance (license, accelerator status, provenance) applies uniformly.

## Reversal conditions

Reverse if a modality family demonstrably cannot meet the shared contracts (e.g. hard dependency on a non-PyTorch runtime) — then split that family into a sidecar package with its own acceptance gate, recorded in a new ADR.
