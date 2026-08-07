# transfer_plan/models/bridges/resampler.md

Source: `medfm/models/bridges/resampler.py` (Perceiver-style static-query token resampler).
Wave: 0.

## Transfer checklist

- [x] `PerceiverResamplerBridge` (fixed queries + `nn.MultiheadAttention` cross-attention +
      zero-row mask safety) → **keep** — transformers exposes `PerceiverModel` (verified) but
      **no standalone `PerceiverResampler`** in this pin, and adopting the full Perceiver model
      would pull a large, differently-shaped architecture over our fixed-bucket contract.
      `nn.MultiheadAttention` inside is already library-native. Keep.


## Result

Verified keep: `transformers.PerceiverModel` exists but `PerceiverResampler` does not, and the
fixed-query, static-bucket, zero-row-safe contract has no drop-in equivalent. Test green:
`uv run --frozen pytest tests/phase_09/test_bridges.py` (4 passed). No source/test files were
modified.