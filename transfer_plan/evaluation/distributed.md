# transfer_plan/evaluation/distributed.md

Source: `medfm/evaluation/distributed.py` (distributed metric reductions, parity contracts).
Wave: 1.

## Transfer checklist

- [x] `reduce_metric_counts` / `reduce_metric_mapping` / `remove_padded_duplicates` →
      **keep** — true-count-based (never rank-mean) reduction is a correctness contract for
      uneven/padded batches; no library (torchmetrics not installed) reproduces remove-padded
      semantics.
- [x] `compare_backend_predictions` / `compare_backend_metrics` / `BackendTolerance` /
      `ParityResult` → **keep** — backend-parity tolerance machine is bespoke; must survive
      metric-math transfers (Wave 0) unchanged so parity gates still hold.
- [x] `gather_host_metadata` / `coordinator_write_report` / `assert_shared_evaluation_seed` →
      **keep** — orchestration glue.

## Tests
`tests/phase_16/test_evaluation.py` (`compare_backend_*` parity).
## Result
Verified keep: true-count reductions, padded-duplicate removal, backend parity tolerance, and coordinator/seed orchestration are bespoke correctness contracts. Source read confirmed the contracts. No parity drift measured. Focused Phase-16 test run — 17 passed.
