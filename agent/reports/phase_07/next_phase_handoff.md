# Phase 07 handoff

- Phase 09 should use `GenericMONAI3DAdapter`/`CTFMAdapter` spatial tokens with `[B,N,D]` ordering flattened depth → height → width. `token_coordinates` are `[B,N,3]` in `MILLIMETERS`, generated from patch-center voxel indices through `SpatialMetadata.affine` (spacing fallback is explicit D,H,W).
- Phase 11/14 task modules can attach a pooled head or consume the ordered feature pyramid `[B,C,D,H,W]`; native NV-Segment-CTMR and MedSAM2 wrappers should bypass generic classification heads for their decoder/lifecycle paths.
- Fixed local contract bucket: generic `(C,16,16,16)` with `(4,4,4)` patches. Declared production adapter buckets are CT `(1,96,96,96)/(16,16,16)`, MRI `(2,64,96,96)/(4,8,8)`, and task-specific MedSAM2 `(1,64,64,64)/(8,8,8)`.
- No unsupported XLA operators or custom CUDA dependencies are used by the generic fallback; native upstream bundle/sequential-memory limitations are recorded in `tpu_smoke_config()` and the unresolved-issues report.
- The accepted 3D path unblocking Phase 09 is the generic/CT-FM-compatible visual token contract; external checkpoint adoption remains a license gate.
