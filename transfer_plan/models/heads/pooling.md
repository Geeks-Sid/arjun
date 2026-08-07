# transfer_plan/models/heads/pooling.md

Source: `medfm/models/heads/pooling.py` (CLS, masked mean, attention, GeM, Top-K, MIL pooling).
Wave: 0.

## Transfer checklist

- [ ] `CLSPooling` / `MaskedMeanPooling` → **keep** — 2–3 line torch reductions; nothing to adopt.
- [ ] `AttentionPooling` / `MILAttentionPooling` (gated attention) → **keep** — TIM's attention
      pooling differs (this is a `softmax(bmm)` over tokens with zero-row safety); torchvision
      has nothing. Hand-rolled is appropriate and tested.
- [ ] `GeneralizedMeanPooling` (GeM, learnable exponent) → **keep** — `timm.layers` has **no**
      GeM in this pin (verified absent); keep the compact implementation.
- [ ] `TopKPooling` → **keep** — no library equivalent.
- [ ] `build_pooling` registry → **keep**.

## Tests
`tests/phase_11/test_heads_and_losses.py`, `tests/phase_13/test_recipes.py`.
