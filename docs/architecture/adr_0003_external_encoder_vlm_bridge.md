# ADR 0003: External-encoder VLM bridge

Status: Accepted (2026-08-04)
Deciders: Project Maintainer

## Context

v1 needs VLM capabilities (VQA, report generation, structured findings) across 2D, 3D, and WSI. Options: train a medical VLM from scratch (out of scope), use a VLM's built-in vision tower only, or bridge external medical encoders into a language model.

## Decision

VLM support is built as an **external-encoder bridge**: a modality-specific `VisualEncoder` produces visual tokens; a trained projector/bridge maps them into the language model's embedding space; the language model (MedGemma 1.5 4B primary) consumes interleaved visual/text tokens via `forward_with_visual_tokens`. MedGemma's native visual pathway remains available as its own adapter for `XRAY_2D`/`MULTI_IMAGE_2D`, but the bridge is the shared mechanism.

## Alternatives considered

- **Native-tower-only (use each VLM as shipped):** locks us to the VLM's own vision tower; cannot use CT-FM/Triad/H-Optimus features. Rejected.
- **Cross-attention (Flamingo-style) bridge:** stronger for some tasks but requires LM-internal surgery per model; higher maintenance. Deferred — projector first.
- **Caption-then-LLM cascade:** loses spatial detail; retrieval/segmentation grounding impossible. Rejected.

## Consequences

- One bridge implementation per (encoder, LM) pair, behind a shared `ProjectedVisualTokens` contract (Phase 09).
- Bridge/projector training is PEFT-scope (ADR 0002) and TPU-feasible with static visual-token buckets (ADR 0008).
- Visual-token coordinate systems must be documented per encoder (Phase 02 contract).

## Reversal conditions

Reverse if projector-only bridging measurably underperforms a cross-attention baseline on the VS-VLM slice acceptance metrics; adopt cross-attention for that pair via a new ADR.
