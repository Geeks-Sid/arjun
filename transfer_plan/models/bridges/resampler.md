# transfer_plan/models/bridges/resampler.md

Source: `medfm/models/bridges/resampler.py` (Perceiver-style static-query token resampler).
Wave: 0.

## Transfer checklist

- [ ] `PerceiverResamplerBridge` (fixed queries + `nn.MultiheadAttention` cross-attention +
      zero-row mask safety) → **keep** — transformers exposes `PerceiverModel` (verified) but
      **no standalone `PerceiverResampler`** in this pin, and adopting the full Perceiver model
      would pull a large, differently-shaped architecture over our fixed-bucket contract.
      `nn.MultiheadAttention` inside is already library-native. Keep.
