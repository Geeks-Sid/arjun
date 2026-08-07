# ADR 0013: Adopt torchmetrics tie-ordering for AUROC/AUPRC and rouge-score for ROUGE-L

Status: Accepted (2026-08-07)
Deciders: Project Maintainer

## Context

ADR-0012 added `torchmetrics`, `scikit-learn`, and `rouge-score` and adopted
the kernels where parity held. Three kernels failed that parity gate and
stayed hand-rolled:

- ranked AUROC / PR-AUC (both the `metrics.py` closed-form rank computation and
  `advanced._average_precision`): torchmetrics' AUROC/AUPRC resolve equal scores
  by sample order, whereas the hand-rolled version used a stable descending
  sort. On tied-score inputs the values differ by up to 0.5 (measured 0.5 for
  AUROC and 0.0278 for AP).
- ROUGE-L: rouge-score's default NLTK WordPunct tokenizer is not guaranteed to
  reproduce the repository's `\w+`-normalized tokenization (hyphens,
  underscores, unicode, contractions), and a token-list parity fixture exposed
  drift of up to 0.6. Using the default would leave ROUGE-L numbers dependent on
  an unrelated tokenizer rather than the repository's own token policy.

The maintainer has approved bumping the contract to complete the migration.

## Decision

1. **AUROC / PR-AUC use torchmetrics.** `metrics._auroc`, `_auprc`, and
   `advanced._average_precision` delegate to `torchmetrics.classification`
   `BinaryAUROC` / `BinaryAveragePrecision`. Tie ordering on equal scores now
   follows torchmetrics (sample order) instead of the previous stable
   descending rank convention; published discrimination scores on tied-score
   inputs may change. The wrapper keeps the undefined-metric contract: a
   single-class sample set still reports `None` (not a fabricated value), and
   the `MetricValue` JSON surface is unchanged. `metrics._sorted_binary` is
   removed (dead after the delegation).
2. **ROUGE-L uses rouge-score with the repository tokenizer preserved.**
   `advanced._rouge_l` delegates to
   `rouge_score.rouge_scorer.RougeScorer(["rougeL"], tokenizer=<repo tokens>)`
   where the custom tokenizer reproduces the existing `_tokens` output exactly
   (normalize + `\w+` findall). This transfers the LCS computation to
   rouge-score while keeping the repository's tokenization, so ROUGE-L numbers
   are unchanged (parity verified to 1e-9 on ordinary and clinical fixtures).
   The default WordPunct tokenizer is deliberately not adopted: exact
   number-for-number parity is achievable for free with the repository
   tokenizer, so no semantic trade is justified.
3. Everything already delegated in ADR-0012 (ECE, `_rank_auc`, MONAI Dice/IoU,
   adjacent-slice) is unchanged. Brier, the histogram calibrator, selective
   risk, and cluster bootstrap remain hand-rolled for the reasons recorded in
   ADR-0012.

## Alternatives considered

- **Keep the stable-descending tie convention:** preserves historic numbers
  but keeps two hand-rolled ranking kernels that reimplement a mature, vetted
  library on the exact path users care about; maintaining bespoke ROC/PR
  interpolation carries its own edge-case risk. Rejected per the approved bump.
- **Pre-sort inputs to force torchmetrics to the repo's tie order:** sorting
  by (score desc, original index) does not change torchmetrics' internal tie
  resolution, which breaks ties by score then input position; no pre-processing
  reproduces the old result without reimplementing the ranking. Rejected.
- **Accept rouge-score WordPunct tokenization:** would make public ROUGE-L
  values depend on a tokenizer that does not reproduce the repository's
  `\w+` policy (drift on hyphen/unicode edge tokens) for no benefit, since a
  custom tokenizer yields an exact number-for-number transfer. Rejected.

## Consequences

- Discrimination metrics on tied-score inputs may differ from prior releases;
  tests asserting `== 1.0` on perfect-separation fixtures are unaffected
  (verified), and the parity tests now assert exact equality with torchmetrics
  on tie fixtures.
- `metrics._auroc`/`_auprc` and `advanced._average_precision` output values
  now match `torchmetrics` command-for-command; consumers that cross-check
  published metrics against torchmetrics will agree exactly.
- ROUGE-L is computed by rouge-score's LCS implementation with the repository
  tokenizer; numbers are unchanged and the dependency is exercised in the
  evaluation path.
- Parity tests in `tests/phase_16/` and `tests/phase_17/` pin the new behavior;
  any future library update that changes tie resolution or ROUGE-L numerics
  surfaces as a numeric test failure.

## Reversal conditions

Reverse or amend if a torchmetrics release changes AUROC/AP tie resolution or
inputs with many ties (e.g. coarse discretized risk scores) show material
ranking instability in practice, or if rouge-score's custom-tokenizer path is
deprecated — adopt via a new ADR with measurements.
