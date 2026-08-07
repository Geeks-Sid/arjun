# Release notes — 0.1.0-rc (research release)

Pre-release research milestone. This software is **not for clinical use** and
makes no diagnostic, treatment, or safety claims. See
`docs/clinical_safety_scope.md` and `docs/release/known_limitations.md`.

## What ships

- Full framework through Phase 17: core contracts, data ingestion/provenance,
  preprocessing (radiology/pathology), model registry + license governance,
  native 3D and pathology adapters, language pillars, LoRA/QLoRA (CUDA) and
  BF16 LoRA (TPU), task modules/losses, training engine (CPU/CUDA/TPU),
  2D/3D/pathology recipes, evaluation/validation, and portable inference
  bundles (NIfTI / reviewed DICOM exports).
- Phase 18 hardening: Level-1 CPU CI, protected hardware workflows (L2-L4,
  CUDA, TPU), golden regression, security/privacy suite, release gate tooling,
  provenance/license validation, and release docs.
- Library-backed kernels (parity-pinned): MONAI metrics/transforms, torchmetrics
  AUROC/PR-AUC/ECE, rouge-score ROUGE-L (ADR-0012/0013).

## Checksums and artifacts

- `docs/release/checksums.txt` — SHA-256 over release docs/artifacts.
- Adapter bundles: built per model; load + smoke validated per bundle.

## Known limitations

All registry models are `UNTESTED` on every accelerator backend until protected
hardware jobs record smoke evidence; several are `BLOCKED` pending license
review. No clinical or safety claims. See the support matrix and known
limitations doc.

## Rollback

Prior release artifacts are immutable (checksummed); rollback = redeploy the
previous bundle per `docs/release/rollback.md`.
