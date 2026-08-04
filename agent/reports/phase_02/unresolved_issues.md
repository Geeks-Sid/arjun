# Phase 02 Unresolved Issues

1. **XLA device-transfer fixture unexecuted on this host.** No TPU is
   available on the development workstation.
   `tests/phase_02/test_device_transfer.py::test_batch_transfer_on_xla` is
   implemented and protected behind `MEDFM_RUN_TPU_TESTS=1`; it must run on a
   Cloud TPU VM (`scripts/tpu_vm_bootstrap.sh`, then `make test-tpu`). Same
   disposition as Phase 01.

2. **Distributed-reduction tests for metric lifecycle are deferred.** The
   `TaskModule` contract documents sufficient-statistics reduction, but a
   multi-rank reduction test needs a multi-device runner
   (`MEDFM_RUN_DISTRIBUTED_TESTS=1`); none exists on this host.

Neither item blocks Phase 03; both are tracked for the TPU/distributed
acceptance runs in later phases.
