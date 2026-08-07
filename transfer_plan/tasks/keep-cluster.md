# transfer_plan/tasks/keep-cluster.md

Covers: `tasks/{classification,segmentation,retrieval,localization,boxes,alignment,generation,
language_segmentation,multitask,reductions,structured,structured_generation,schemas,base}.py`.
Wave: 1/2 — **keep by design** (thin task wrappers over heads/losses; reduction/count contract).

## Transfer checklist

- [x] Task wrappers (`ClassificationTask`, `BinarySegmentationTask`, `RetrievalTask`,
      `LocalizationTask`, `StructuredGenerationTask`, ...) → **keep** — they bind heads +
      losses to `MedicalBatch` targets and produce `LossOutput` with valid-sample counts;
      replacing them with a library task framework has no counterpart. Their numeric core
      (losses, boxes) transfers in Wave 0 (see `tasks/losses.md`, `models/heads/localization.md`).
- [x] `reductions.py` (`reduce_mean_by_count`, `reduce_loss_output`) → **keep** —
      true-count reduction over pads is a distributed-training correctness contract; no library.
- [x] `structured.py` / `structured_generation.py` / `schemas.py` → **keep** — jsonschema-backed
      structured-findings validation; already library-based.
- [x] `base.py` (`TaskModuleBase`, count/mask helpers) → **keep** — contract.
 
## Result

Verified keep by design. Quick source reads confirmed task wrappers bind heads/losses to `MedicalBatch` and emit `LossOutput` counts; reductions preserve true-count distributed semantics; structured validation is jsonschema-backed; and `TaskModuleBase` owns contract helpers. No source or test files changed. Parity drift: not applicable (keeps).

Focused verification: `uv run --frozen pytest tests/phase_11/test_task_wrappers.py tests/phase_11/test_segmentation.py tests/phase_11/test_alignment_boxes_generation.py tests/phase_09/test_bridges.py tests/phase_08/test_pathology.py tests/phase_15/test_recipes.py` — 61 passed.

## Tests
`tests/phase_11/test_task_wrappers.py`, `test_segmentation.py`, `test_alignment_boxes_generation.py`,
`tests/phase_13/test_recipes.py`.
