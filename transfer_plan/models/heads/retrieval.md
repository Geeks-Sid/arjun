# transfer_plan/models/heads/retrieval.md

Source: `medfm/models/heads/retrieval.py` (image/text projection + symmetric contrastive loss).
Wave: 0.

## Transfer checklist

- [ ] `symmetric_contrastive_loss` / `SymmetricContrastiveLoss` → **keep** — this is a
      CLIP-style symmetric InfoNCE with **same-patient negative masking** (`_patient_validity`).
      MONAI's `ContrastiveLoss(temperature, batch_size)` is a different contrastive formulation
      (requires labels, no per-sample validity), so no drop-in; hand-rolled is correct here.
      The projection heads are `nn.Linear`+`F.normalize` glue — nothing to adopt.
- [ ] `ImageTextProjectionHead` / `ImageTextRetrievalHead` → **keep** — network glue.
- [ ] `DistributedNegativeProvider` protocol → **keep** (future boundary).

## Tests
`tests/phase_11/test_alignment_boxes_generation.py`.
