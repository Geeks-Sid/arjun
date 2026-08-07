# transfer_plan/tasks/keep-cluster.md

Covers: `tasks/{classification,segmentation,retrieval,localization,boxes,alignment,generation,
language_segmentation,multitask,reductions,structured,structured_generation,schemas,base}.py`.
Wave: 1/2 — **keep by design** (thin task wrappers over heads/losses; reduction/count contract).

## Transfer checklist

- [ ] Task wrappers (`ClassificationTask`, `BinarySegmentationTask`, `RetrievalTask`,
      `LocalizationTask`, `StructuredGenerationTask`, ...) → **keep** — they bind heads +
      losses to `MedicalBatch` targets and produce `LossOutput` with valid-sample counts;
      replacing them with a library task framework has no counterpart. Their numeric core
      (losses, boxes) transfers in Wave 0 (see `tasks/losses.md`, `models/heads/localization.md`).
- [ ] `reductions.py` (`reduce_mean_by_count`, `reduce_loss_output`) → **keep** —
      true-count reduction over pads is a distributed-training correctness contract; no library.
- [ ] `structured.py` / `structured_generation.py` / `schemas.py` → **keep** — jsonschema-backed
      structured-findings validation; already library-based.
- [ ] `base.py` (`TaskModuleBase`, count/mask helpers) → **keep** — contract.

## Tests
`tests/phase_11/test_task_wrappers.py`, `test_segmentation.py`, `test_alignment_boxes_generation.py`,
`tests/phase_13/test_recipes.py`.
