from __future__ import annotations

import pytest

from medfm.evaluation import advanced

# String-level fixtures: the repo `_rouge_l` maps (predicted, reference) text to
# ROUGE-L recall over the repository `_tokens` (\w+ normalized) tokenization.
_ROUGE_STRING_PAIRS = (
    ("the cat sat on the mat", "the cat sat on the mat", 1.0),
    ("left lung opacity", "left lung", 1.0),
    ("no acute findings", "acute findings in left lower lobe", 1 / 3),
    ("left-lower lobe", "left lower lobe", 1.0),
    ("", "left lower lobe", 0.0),
    ("left lower lobe", "", 0.0),
)


def _string_lcs_recall(predicted: str, target: str) -> float:
    """Reference: in-repo LCS DP over ``_tokens`` (the pre-transfer kernel)."""
    left, right = advanced._tokens(predicted), advanced._tokens(target)
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for row, token in enumerate(left, start=1):
        for column, other in enumerate(right, start=1):
            table[row][column] = (
                table[row - 1][column - 1] + 1
                if token == other
                else max(table[row - 1][column], table[row][column - 1])
            )
    return 0.0 if not right else table[-1][-1] / len(right)


@pytest.mark.parametrize(("predicted", "target", "expected"), _ROUGE_STRING_PAIRS)
def test_rouge_l_preserves_previous_string_kernel(predicted: str, target: str, expected: float) -> None:
    # ADR-0013: _rouge_l now delegates the LCS to rouge-score with the repo
    # tokenizer; recall must be number-for-number identical to the prior DP.
    assert _string_lcs_recall(predicted, target) == pytest.approx(expected, abs=1e-6)
    assert advanced._rouge_l(predicted, target) == pytest.approx(expected, abs=1e-6)


def test_rouge_l_uses_repository_tokenization_not_wordpunct() -> None:
    # The transfer pinned rouge-score to the repo tokenizer; the default NLTK
    # WordPunct tokenizer is deliberately not used, so no tokenizer drift can
    # enter published ROUGE-L values.
    from rouge_score import rouge_scorer

    predicted, target = "left-lower lobe", "left lower lobe"
    default = rouge_scorer.RougeScorer(["rougeL"]).score(target, predicted)["rougeL"]
    repo = _string_lcs_recall(predicted, target)
    assert default.recall == pytest.approx(repo, abs=1e-6)
    assert advanced._rouge_l(predicted, target) == pytest.approx(repo, abs=1e-6)
