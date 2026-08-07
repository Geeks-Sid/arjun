# Phase 07: Native 3D Visual-Encoder Adapters

Implemented the native-volume contract and offline pure-PyTorch fallback path.

## Delivered

- `GenericMONAI3DAdapter` consumes canonical `[B,C,D,H,W]` volumes without slice folding.
- Fixed patch grids emit pooled embeddings, spatial tokens, feature maps, masks, and physical token coordinates when affine/spacing metadata is present.
- Metadata validation covers shape, orientation, spacing, channel/sequence declaration, and dtype.
- Host-side fixed-window sliding inference reconstructs dense outputs by overlap averaging.
- CTFM, FlexiCT 3D/VLM identities, Triad MAE/SimMIM identities, NV-Segment-CTMR, MedSAM2, Merlin, M3D-CLIP/LaMed are separately declared and registry-gated.
- LoRA targets are limited to transformer attention/MLP projections; patch convolutions remain excluded.
- Checkpoint round-trip, adapter-only export, TPU fixed-shape configuration, and native task lifecycles are covered offline.

## Verification

- `uv run pytest tests/phase_07 -q`: 13 passed.
- `uv run pytest tests/phase_05 tests/phase_06 -q`: 134 passed.
- `uv run python -m medfm.tools.smoke --phase 07 --json`: passed.
- `uv run ruff check` on all Phase 07 Python files: passed.

Real upstream weights remain blocked by the unresolved license/checkpoint records already present in `model_registry/licenses.yaml`; no upstream capability is claimed by the local fallback adapters.
