# transfer_plan/evaluation/distributed.md

Source: `medfm/evaluation/distributed.py` (distributed metric reductions, parity contracts).
Wave: 1.

## Transfer checklist

- [ ] `reduce_metric_counts` / `reduce_metric_mapping` / `remove_padded_duplicates` →
      **keep** — true-count-based (never rank-mean) reduction is a correctness contract for
      uneven/padded batches; no library (torchmetrics not installed) reproduces remove-padded
      semantics.
- [ ] `compare_backend_predictions` / `compare_backend_metrics` / `BackendTolerance` /
      `ParityResult` → **keep** — backend-parity tolerance machine is bespoke; must survive
      metric-math transfers (Wave 0) unchanged so parity gates still hold.
- [ ] `gather_host_metadata` / `coordinator_write_report` / `assert_shared_evaluation_seed` →
      **keep** — orchestration glue.

## Tests
`tests/phase_16/test_evaluation.py` (`compare_backend_*` parity).
