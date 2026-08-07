# transfer_plan/models/heads/pooling.md

Source: `medfm/models/heads/pooling.py` (CLS, masked mean, attention, GeM, Top-K, MIL pooling).
Wave: 0.

## Transfer checklist

- [x] `CLSPooling` / `MaskedMeanPooling` → **keep** — 2–3 line torch reductions; nothing to adopt.
- [x] `AttentionPooling` / `MILAttentionPooling` (gated attention) → **keep** — TIM's attention
      pooling differs (this is a `softmax(bmm)` over tokens with zero-row safety); torchvision
      has nothing. Hand-rolled is appropriate and tested.
- [x] `GeneralizedMeanPooling` (GeM, learnable exponent) → **keep** — `timm.layers` has **no**
      GeM in this pin (verified absent); keep the compact implementation.
- [x] `TopKPooling` → **keep** — no library equivalent.
- [x] `build_pooling` registry → **keep**.

## Tests
`tests/phase_11/test_heads_and_losses.py`, `tests/phase_13/test_recipes.py`.


## Result

Verified keep for all five items: reductions, mask-safe attention, learnable GeM, Top-K,
and the stable registry are compact contract logic with no matching installed drop-in.
Tests green: `uv run --frozen pytest tests/phase_11/test_heads_and_losses.py` and
`uv run --frozen pytest tests/phase_13/test_recipes.py` (20 passed total). No source/test
files were modified.