# transfer_plan/models/heads/retrieval.md

Source: `medfm/models/heads/retrieval.py` (image/text projection + symmetric contrastive loss).
Wave: 0.

## Transfer checklist

- [x] `symmetric_contrastive_loss` / `SymmetricContrastiveLoss` → **keep** — this is a
      CLIP-style symmetric InfoNCE with **same-patient negative masking** (`_patient_validity`).
      MONAI's `ContrastiveLoss(temperature, batch_size)` is a different contrastive formulation
      (requires labels, no per-sample validity), so no drop-in; hand-rolled is correct here.
      The projection heads are `nn.Linear`+`F.normalize` glue — nothing to adopt.
- [x] `ImageTextProjectionHead` / `ImageTextRetrievalHead` → **keep** — network glue.
- [x] `DistributedNegativeProvider` protocol → **keep** (future boundary).

## Tests
`tests/phase_11/test_alignment_boxes_generation.py`.

## Result

Verified keep for all three items: MONAI's contrastive loss does not preserve the CLIP-style
same-patient negative mask, while projection and distributed-negative pieces are contract glue.
Tests green: `uv run --frozen pytest tests/phase_11/test_alignment_boxes_generation.py`
(6 passed). No source/test files were modified.
