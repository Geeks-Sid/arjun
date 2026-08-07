# Unresolved Issues — Phase 06

## 1. MedSigLIP real-checkpoint smoke
- **Status:** Blocked by gated license (HAI-DEF terms, named-individual acceptance required).
- **Owner:** Project Maintainer (Siddhesh)
- **Rev:** 9cea28a1a1195f665105faa6e8544c112fd960a4
- **Action:** Accept HAI-DEF terms at https://huggingface.co/google/medsiglip-448, then `medfm models accept-terms medsiglip --by <name>`, then run real-checkpoint GPU tests.

## 2. H-Optimus-0 real-checkpoint acceptance
- **Status:** Blocked by gated license (Bioptimus, Apache-2.0) + timm pretrained load.
- **Owner:** Project Maintainer (Siddhesh) / Phase 08
- **Rev:** b145cc1e6c6b30d3251aa8b1f844e6974188a743
- **Action:** Accept terms at https://huggingface.co/bioptimus/H-optimus-0, install timm with HF token, run GPU smoke. Handed to Phase 08 for WSI/MIL integration.

## 3. MedGemma vision real-checkpoint acceptance
- **Status:** Blocked by gated license (HAI-DEF terms).
- **Owner:** Project Maintainer (Siddhesh)
- **Rev:** 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b
- **Action:** Accept HAI-DEF terms, run GPU smoke.

## 4. TPU smoke evidence
- **Status:** No TPU runtime available locally. All TPU backend_support stays UNTESTED.
- **Action:** Run tpu_smoke_config + real XLA forward when TPU VM is available (Phase 12/CI).

## 5. CUDA real-checkpoint VRAM measurement
- **Status:** Not yet recorded for Phase 06 models.
- **Action:** Run GPU-protected tests with MEDFM_RUN_GPU_TESTS=1.
