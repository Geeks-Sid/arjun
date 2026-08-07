# transfer_plan/training/optimizer.md

Source: `medfm/training/optimizer.py` (optimizer construction, staged freezing, audits).

Wave: 0.

## Transfer checklist

- [x] `_make_optimizer` → **keep** — it already passes through to `torch.optim`
      (AdamW/Adam, `fused`/`foreach`), or to bitsandbytes `optim.AdamW8bit` on the CUDA path.
      Library-backed; no transfer.
- [x] `_make_scheduler` (warmup + cosine/linear `LambdaLR`) → **keep** — it already
      returns `torch.optim.lr_scheduler.LambdaLR` with a hand-computed schedule fn. Native
      `LinearLR`/`CosineAnnealingLR`/`SequentialLR` replacement is optional, but the staged-freeze
      schedule (applied at boundaries by `rebuild_optimizer`) couples to our LambdaLR. Keep the
      torch-native lambda; LR is a Python float and has no dtype concern.
- [x] `build_parameter_groups` / `_group_hyperparameters` → **keep** — per-role LR/weight-decay
      grouping over `ROLE_ORDER` is recipe glue; no library equivalent.
- [x] `apply_freeze_schedule` / `rebuild_optimizer` / `audit_gradients` / `TrainabilityAuditError`
      → **keep** — staged-freeze + param-group rebuild is contract logic.
- [x] `canonical_role` / `role_for_parameter` → **keep**.


## Tests
`tests/phase_10/test_config_and_backend.py`, `tests/phase_12/test_trainer_memory_checkpoint.py`,
`tests/phase_12/test_config_backend.py`.

## Result

Verified keep for all five items. Optimizer construction already delegates to torch (or the
guarded CUDA-only bitsandbytes path), and the LambdaLR schedule remains intentionally
torch-native; no native scheduler swap or extra parity harness was required. Tests green:
`uv run --frozen pytest tests/phase_10/test_config_and_backend.py` (5 passed) and
`uv run --frozen pytest tests/phase_12/test_trainer_memory_checkpoint.py
tests/phase_12/test_config_backend.py` (12 passed). No source/test files were modified.
