# transfer_plan/models/pathology/selection.md

Source: `medfm/models/pathology/selection.py` (deterministic WSI tile selection + fixed visual
token selection).
Wave: 1.

## Transfer checklist

- [x] `GridTileSampler` / `RandomTileSampler` / `QualityWeightedTileSampler` →
      `DiversityTileSampler` (greedy farthest-point) / `TopKAttentionTileSampler` /
      `TextConditionedTileSampler` → **keep** — tile selection with deterministic seeded keys
      (SHA-256 ordering) and fixed `TokenBudget` (static-shape) has no off-the-shelf library;
      the greedy farthest-point selection is bespoke (GigaPath-style). Keep the hand-rolled
      determinism (test-pinned).
- [x] `WSITokenSelector` / `SelectedTokens` → **keep** — fixed-width padding + mask + evidence
      is contract.
- [x] External reference: prov-gigapath's tile encoder/sampling (Apache-2.0) may inform a
      future `DiversityTileSampler` tweak, but only as a documented borrow (see README §1.7).

## Result

Verified keep by design. Quick source read confirmed deterministic grid/random/quality samplers, bespoke greedy farthest-point and attention/text-conditioned selection, and fixed-width `TokenBudget`/`SelectedTokens` padding-mask-evidence contracts. No source or test files changed. Parity drift: not applicable (keep).

Focused verification: `uv run --frozen pytest tests/phase_08/test_pathology.py tests/phase_15/test_recipes.py` — 41 passed.

## Tests
`tests/phase_08/test_pathology.py`, `tests/phase_15`-recipe tests.
