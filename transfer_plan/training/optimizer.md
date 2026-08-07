# transfer_plan/training/optimizer.md

Source: `medfm/training/optimizer.py` (optimizer construction, staged freezing, audits).

Wave: 0.

## Transfer checklist

- [ ] `_make_optimizer` → **keep** — it already passes through to `torch.optim`
      (AdamW/Adam, `fused`/`foreach`), or to bitsandbytes `optim.AdamW8bit` on the CUDA path.
      Library-backed; no transfer.
- [ ] `_make_scheduler` (warmup + cosine/linear `LambdaLR`) → **partial/keep** — it already
      returns `torch.optim.lr_scheduler.LambdaLR` with a hand-computed schedule fn. torch's
      native `LinearLR`/`CosineAnnealingLR`/`SequentialLR` could replace the lambda, but the
      staged-freeze schedule (applied at boundaries by `rebuild_optimizer`) couples to our
      LambdaLR. Verify a swap to `torch.optim.lr_scheduler` native classes yields identical LR
      values at every step via a unit test; otherwise keep the lambda (it is already torch).
      Zero dtype concern (LR is a Python float).
- [ ] `build_parameter_groups` / `_group_hyperparameters` → **keep** — per-role LR/weight-decay
      grouping over `ROLE_ORDER` is recipe glue; no library equivalent.
- [ ] `apply_freeze_schedule` / `rebuild_optimizer` / `audit_gradients` / `TrainabilityAuditError`
      → **keep** — staged-freeze + param-group rebuild is contract logic.
- [ ] `canonical_role` / `role_for_parameter` → **keep**.

## Tests
`tests/phase_10/test_config_and_backend.py`, `tests/phase_12/test_trainer_memory_checkpoint.py`,
`tests/phase_12/test_config_backend.py`.
