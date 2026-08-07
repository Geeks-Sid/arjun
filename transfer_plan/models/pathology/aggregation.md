# transfer_plan/models/pathology/aggregation.md

Source: `medfm/models/pathology/aggregation.py` (slide-level aggregators: mean, attention-MIL).
Wave: 0.

## Transfer checklist

- [x] `MeanPoolingAggregator` → **keep** — masked mean pooling, a few lines of torch.
- [x] `AttentionMILAggregator` (gated attention MIL with mask-safe softmax) → **keep (code
      provenance option)** — ABMIL is a well-known architecture; no *installed* library exposes
      it (MONAI has no slide aggregator; torch/timm none). Where our implementation lacks a
      feature (e.g. a specific gating variant), we may **borrow the reference implementation**
      from `external_repos/prov-gigapath` (Apache-2.0, commercial-safe) or the CONCH/tools
      utilities (CC-BY-NC-ND — research-only, keep out of commercial bundles; see
      `docs/licensing_policy.md`). Both repos implement gated-attention MIL; if reusing,
      vendor the file with the upstream license header intact and add attribution in
      `model_registry/`. This is a **code-copy** rather than a library transfer — trigger only
      if a gap is found; otherwise keep ours.
- [x] `GigaPathFlashAggregator` / `TITANAggregator` (identity subclasses with evidence
      metadata) → **keep** — thin boundaries; the native Flash/TITAN weights are optional and
      not wired here.

## Tests
`tests/phase_08/test_pathology.py`, `tests/phase_15`-recipe tests.

## Result

Verified keep for all three items: masked reductions and gated attention are compact contract
logic, no installed slide-level MIL drop-in exists, and the native Flash/TITAN weights remain
optional boundaries. Tests green: `uv run --frozen pytest tests/phase_08/test_pathology.py
tests/phase_15/test_recipes.py` (41 passed). No source/test files were modified.
