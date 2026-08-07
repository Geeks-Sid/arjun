from __future__ import annotations

import pytest
from rouge_score.rouge_scorer import RougeScorer

from medfm.evaluation.advanced import _rouge_l

ROUGE_FIXTURES = (
    (
        "ordinary_tokens",
        ["left", "lung", "opacity"],
        ["left", "lung"],
        (2 / 3, 1.0, 0.8),
        True,
    ),
    (
        "clinical_punctuation_tokens",
        ["left-lower", "lobe"],
        ["left", "lower", "lobe"],
        (0.5, 1 / 3, 0.4),
        False,
    ),
)


def _lcs_length(predicted: list[str], reference: list[str]) -> int:
    table = [[0] * (len(reference) + 1) for _ in range(len(predicted) + 1)]
    for row, token in enumerate(predicted, start=1):
        for column, other in enumerate(reference, start=1):
            table[row][column] = (
                table[row - 1][column - 1] + 1
                if token == other
                else max(table[row - 1][column], table[row][column - 1])
            )
    return table[-1][-1]


def _repo_rouge_stats(predicted: list[str], reference: list[str]) -> tuple[float, float, float]:
    overlap = _lcs_length(predicted, reference)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(reference) if reference else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, fmeasure


@pytest.mark.parametrize(("name", "predicted", "reference", "expected", "parity"), ROUGE_FIXTURES)
def test_rouge_l_tokenization_parity_gate(
    name: str,
    predicted: list[str],
    reference: list[str],
    expected: tuple[float, float, float],
    parity: bool,
) -> None:
    del name
    actual = _repo_rouge_stats(predicted, reference)
    score = RougeScorer(["rougeL"]).score(" ".join(reference), " ".join(predicted))["rougeL"]
    library = (score.precision, score.recall, score.fmeasure)
    kernel_recall = _rouge_l(" ".join(predicted), " ".join(reference))
    if parity:
        assert library == pytest.approx(expected, abs=1e-6)
        assert library == pytest.approx(actual, abs=1e-6)
        assert kernel_recall == pytest.approx(actual[1], abs=1e-6)
    else:
        assert actual == pytest.approx(expected, abs=1e-6)
        assert library == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)
        assert max(abs(left - right) for left, right in zip(library, actual, strict=True)) > 1e-6
        assert abs(kernel_recall - actual[1]) > 1e-6
