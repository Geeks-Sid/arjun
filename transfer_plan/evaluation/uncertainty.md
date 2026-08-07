# transfer_plan/evaluation/uncertainty.md

Source: `medfm/evaluation/uncertainty.py` (entropy/confidence + selective-risk curves).
Wave: 1.

## Transfer checklist

- [x] `uncertainty_metrics` (mean_entropy, mean_confidence, selective-risk@coverage) → **keep** —
      entropy of binary probs is a 2-line torch expression; selective-risk curves over
      low-uncertainty retention have no installed library (torchmetrics not installed; the
      concept is thin, deterministic code already). Keep.
- [x] `_probabilities` (logit→prob auto-detect, finite check) → **keep**.
## Result
Verified keep: finite/logit probability handling, binary entropy/confidence, and selective-risk retention semantics are deterministic custom logic with no installed drop-in library. Source read confirmed the contract. No parity drift measured. Focused Phase-16 test run — 17 passed.
