# Phase 04 Unresolved Issues

1. **CUDA/TPU timing tests are protected-hardware skips.** Host-vs-device
   preprocessing wait is instrumented (`medfm/data/transforms/timing.py`) and
   exercised on CPU; the CUDA and TPU measurement tests are marked
   `gpu`/`tpu` and only run on protected runners
   (`MEDFM_RUN_GPU_TESTS=1` / `MEDFM_RUN_TPU_TESTS=1`). Independent host/device
   measurements on real accelerators must be collected on those runners before
   Phase 12 performance acceptance.
2. **Stochastic 2D spatial augmentations are not invertible.** Rotation /
   translation / scale / flip augmentation records parameters but registers
   no inverter (augmentation is train-time only and never part of an
   evaluation history). If augmented-space evaluation inversion is ever
   needed, affine-parameterized inverters must be added. Intensity/noise and
   stain augmentation are non-invertible by design and skipped by
   `invert_history` (`strict=True` surfaces them).
3. **Resample inversion is shape-exact but value-approximate.** Zoom-then-
   unzoom restores the original lattice and label masks stay discrete
   (nearest), but continuous image values are reconstructed by interpolation
   and are not bit-exact; tolerance-based comparison is required (masks must
   be compared in original coordinates as the phase tests do).
4. **Bias-field correction is a low-pass log approximation**, not full
   SimpleITK N4. It is explicit/offline-only by design; wire real N4 offline
   if exact correction is required.
5. **Stain normalization is Reinhard (mean/std matching)**, not Macenko.
   Reference statistics are caller configuration; a scanner-calibrated
   reference set is follow-up work.
6. **WSI tile planning operates on in-memory arrays.** Slide I/O stays in the
   Phase 03 reader layer; integrating `plan_tiles` with the OpenSlide/
   TiffSlide backends (region reads per `TileRecord`) is the Phase 08 adapter
   handoff.
7. **MULTI_SERIES_3D has no dedicated BucketKind** (Phase 02 vocabulary
   predates it); series counts pad to the per-batch max or reuse a declared
   MULTI_IMAGE count bucket. Add a dedicated kind via contract change if
   static series counts become necessary.
