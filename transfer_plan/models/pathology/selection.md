# transfer_plan/models/pathology/selection.md

Source: `medfm/models/pathology/selection.py` (deterministic WSI tile selection + fixed visual
token selection).
Wave: 1.

## Transfer checklist

- [ ] `GridTileSampler` / `RandomTileSampler` / `QualityWeightedTileSampler` →
      `DiversityTileSampler` (greedy farthest-point) / `TopKAttentionTileSampler` /
      `TextConditionedTileSampler` → **keep** — tile selection with deterministic seeded keys
      (SHA-256 ordering) and fixed `TokenBudget` (static-shape) has no off-the-shelf library;
      the greedy farthest-point selection is bespoke (GigaPath-style). Keep the hand-rolled
      determinism (test-pinned).
- [ ] `WSITokenSelector` / `SelectedTokens` → **keep** — fixed-width padding + mask + evidence
      is contract.
- [ ] External reference: prov-gigapath's tile encoder/sampling (Apache-2.0) may inform a
      future `DiversityTileSampler` tweak, but only as a documented borrow (see README §1.7).

## Tests
`tests/phase_08/test_pathology.py`, `tests/phase_15`-recipe tests.
