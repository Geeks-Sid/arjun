# ADR 0012: Core evaluation metrics via torchmetrics / scikit-learn / rouge-score

Status: Accepted (2026-08-07)
Deciders: Project Maintainer

## Context

The open-source transfer plan (`transfer_plan/`) keeps several clinical
evaluation kernels hand-rolled solely because the mature library was *not
installed*: ranked AUROC, PR-AUC, expected calibration error, and ROUGE-L.
The transfer ground rules require parity-first adoption with zero behavior
drift, and the plan explicitly noted these were out of scope without an ADR.
Evaluation is core runtime (every install profile includes the `medical`
extra), so making these available locally is a dependency-policy change, not
just a code change.

## Decision

1. Add three permissively licensed libraries to the `medical` extra so every
   standard install (`make install`, `make install-tpu`) ships them:
   `torchmetrics>=1.4` (Apache-2.0), `scikit-learn>=1.5` (BSD-3), and
   `rouge-score>=0.1.2` (Apache-2.0). None have CUDA-only or TPU-hostile
   transitive dependencies; all are CPU-safe.
2. Delegate the following kernels to libraries **only where a parity test
   proves identical numerics on the repo's contract** (padded-unpadded
   evaluation flow, float dtype in play, deterministic seeding), mirroring the
   MONAI/torchvision transfers:
   - ranked AUROC / PR-AUC → `torchmetrics` (`BinaryAUROC`,
     `BinaryAveragePrecision`);
   - expected calibration error (ECE) → `torchmetrics` (`BinaryCalibrationError`)
     where binning/norm semantics match.
   `torchmetrics` 1.9 no longer ships a Brier score, so `brier_score` stays a
   hand-rolled torch op. The piecewise-constant histogram calibrator (test-pinned
   determinism) stays as-is; `sklearn.IsotonicRegression` is available but is a
   *different* fit, not a drop-in replacement, and is not adopted in this ADR.
3. ROUGE-L in `generation_metrics` → `rouge-score` only if the tokenized-input
   parity holds (its NLTK WordPunct tokenizer re-tokenizes joined tokens, so
   parity is measured, not assumed); otherwise stay on the token-table DP.
4. Empty-mask / empty-coverage conventions, mask-aware reductions, and the
   `MetricValue` JSON surface are unchanged; transfers happen inside kernels.

## Alternatives considered

- **Keep everything hand-rolled:** zero new deps, but 2-3 metric kernels
  reimplement mature, GPU-friendly libraries (AUROC/PR-AUC/ECE) with no
  contract benefit. Rejected.
- **torchmetrics as a separate `eval` extra:** true, but Evaluation runs in
  every profile (training recipes evaluate continuously); a separate extra would
  make the base install fail to run `evaluation.metrics`. The `medical` home
  matches `evaluation` living under the medical domain. Rejected.
- **Adopt `sklearn.IsotonicRegression` as the calibrator:** a different
  calibration algorithm with its own fit semantics; swapping would change
  threshold-selection and ECE behavior that tests pin. Rejected for this ADR;
  available as a future opt-in.
- **Adopt `statsmodels` for bootstrap CIs:** not requested; cluster bootstrap
  with deterministic seeding has no direct equivalent. Rejected.

## Consequences

- `make install` / `install-tpu` now resolve `torchmetrics`, `scikit-learn`,
  and `rouge-score`; lockfile updated.
- `evaluation.*` and `generation_metrics` may import these libraries at
  module top-level (they are present in every supported install profile).
- Parity tests in `tests/phase_16/` / `tests/phase_17/` pin the delegated
  kernels; any future upstream drift that breaks a metric is caught as a
  numeric test failure rather than silently changing published numbers.
- Hand-rolled kernels that have no library equivalent (Brier, histogram
  calibrator, selective-risk curves, cluster bootstrap CI) remain and are
  documented as `keep` in their transfer checklists.

## Reversal conditions

Reverse or amend if a library update changes metric semantics (e.g. a
torchmetrics release that relocates or alters `CalibrationError` binning), if
rouge-score's NLTK dependency becomes a reproducibility problem in an offline
CI, or if profiling shows the library call path measurably regresses
throughput on the distributed evaluation loop — adopt via a new ADR with
measurements.
