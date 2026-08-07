# transfer_plan/evaluation/uncertainty.md

Source: `medfm/evaluation/uncertainty.py` (entropy/confidence + selective-risk curves).
Wave: 1.

## Transfer checklist

- [ ] `uncertainty_metrics` (mean_entropy, mean_confidence, selective_risk@coverage) → **keep** —
      entropy of binary probs is a 2-line torch expression; selective-risk curves over
      low-uncertainty retention have no installed library (torchmetrics not installed; the
      concept is thin, deterministic code already). Keep.
- [ ] `_probabilities` (logit→prob auto-detect, finite check) → **keep**.
