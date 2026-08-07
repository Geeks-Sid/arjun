# Next phase handoff

Phase 17 implementation and focused acceptance gate are complete.

Recommended Phase 18 follow-up:

1. Add the Phase 17 acceptance command to CI together with the focused
   regression suite and machine validation of the published request/response
   and bundle schemas.
2. Install the medical optional dependencies in a clean validation job and run
   an approved synthetic DICOM SEG workflow, including source-reference and
   geometry reopen checks.
3. Run the same portable adapter bundle on available CPU, CUDA, and TPU/XLA
   workers. Record output tolerances, cold/warm latency, peak memory, bucket
   compile counts, host synchronization, and steady-state latency separately.
4. Populate the governed license catalog and blocked-bundle registry before
   registering any real model; retain the review record with bundle checksums.
5. Exercise timeout cancellation, concurrent adapter switching, and worker
   isolation under representative load. Keep raw reports/images/UIDs in the
   access-controlled clinical store only.
6. Compare exported adapter outputs against training-repository fixtures and
   retain the preprocessing/postprocessing/config hashes with the release
   artifact.

Phase 17 artifacts are under `agent/reports/phase_17/`; published deployment
contracts are under `docs/inference/`.
