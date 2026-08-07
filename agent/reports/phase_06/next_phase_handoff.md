# Phase 06 → Phase 07+ Handoff

## Published adapter output dimensions

| Adapter | Pooled | Spatial tokens | Feature maps | Hidden state layers |
|---|---|---|---|---|
| MedSigLIP | [B, 1152] | [B, 1024, 1152] | [B, 1152, 32, 32] × 4 | 27 layers |
| RAD-DINO | [B, 768] | [B, 1369, 768] | [B, 768, 37, 37] × 4 | 12 layers (3,6,9,12) |
| H-Optimus-0 | [B, 1536] | [B, 256, 1536] | [B, 1536, 16, 16] × 4 | 40 layers (10,20,30,39) |
| MedGemma vision | [B, lm_dim] | [B, 256, lm_dim] | unavailable | — |

## Token coordinate semantics
All 2D adapters: NORMALIZED_IMAGE (x, y ∈ [0, 1]), patch-center grid, row-major.

## Unblocks Phase 09
RAD-DINO (READY, MIT) unblocks Phase 09 bridge/VLM work immediately.

## Revision-sensitive hooks
- RAD-DINO feature_map_layers: pinned to 12-layer tower, hidden_states[3,6,9,12]
- MedSigLIP feature_map_layers: pinned to 27-layer tower
- H-Optimus intermediate layers: pinned to 40-layer tower
If upstream revision changes depth, the adapter must be re-verified.

## Custom kernels and fallback paths
- All backbones use PyTorch SDPA (config._attn_implementation="sdpa")
- No CUDA custom kernels registered
- pure_pytorch_fallback=True for all adapters

## Backend parity tolerances
- FP32: bit-exact (same RNG seed)
- BF16: tolerance TBD after real-checkpoint smoke on GPU + TPU
