"""Pattern-based PHI screening for clinical text.

This is a *screening* layer, not a de-identification guarantee: it flags
common identifier-shaped patterns (MRN-like numbers, dates, phone numbers,
emails, SSN-like numbers) so a policy can warn or fail the sample. All
patterns are configurable; sites are expected to extend them via
``extra_patterns``.

Privacy rule (docs/data_governance.md): :class:`PHIMatch` carries only the
category and character offsets — NEVER the matched text. Logging and error
messages in this module contain counts per category only.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from medfm.data.errors import TextPreprocessError

_logger = logging.getLogger(__name__)

#: Default screening patterns, compiled case-insensitively. These are
#: deliberately conservative shape heuristics, not validators.
_DEFAULT_PATTERNS: dict[str, str] = {
    # Explicitly labeled MRN, or any standalone run of 7+ digits.
    "mrn": r"(?:\bMRN\s*[:#]?\s*[A-Za-z0-9-]{4,}\b)|(?:\b\d{7,}\b)",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "date": r"(?:\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)|(?:\b\d{4}-\d{2}-\d{2}\b)",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
}


@dataclass(frozen=True)
class PHIMatch:
    """One screening hit. Carries offsets only — never the matched text."""

    category: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.category:
            raise TextPreprocessError("PHIMatch.category must be non-empty")
        if self.start < 0 or self.end < self.start:
            raise TextPreprocessError(
                f"PHIMatch offsets must satisfy 0 <= start <= end; got ({self.start}, {self.end})"
            )


def phi_scan(text: str, extra_patterns: dict[str, str] | None = None) -> list[PHIMatch]:
    """Scan ``text`` for PHI-shaped patterns; return matches sorted by offset.

    ``extra_patterns`` maps a category name to a regex string; a category
    present in both defaults and ``extra_patterns`` uses the caller's
    pattern. Invalid regexes raise :class:`TextPreprocessError` (the message
    names the category, never the text).
    """
    patterns: dict[str, str] = dict(_DEFAULT_PATTERNS)
    if extra_patterns:
        patterns.update(extra_patterns)
    compiled: dict[str, re.Pattern[str]] = {}
    for category, pattern in patterns.items():
        try:
            compiled[category] = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise TextPreprocessError(f"invalid PHI regex for category {category!r}: {exc}") from exc
    matches = [
        PHIMatch(category=category, start=m.start(), end=m.end())
        for category, regex in compiled.items()
        for m in regex.finditer(text)
    ]
    matches.sort(key=lambda m: (m.start, m.end, m.category))
    return matches


@dataclass(frozen=True)
class PHIPolicy:
    """How to react to PHI screening hits.

    ``mode``:
    - ``"off"`` — skip scanning entirely.
    - ``"warn"`` — log a warning with counts per category (never text).
    - ``"error"`` — raise :class:`TextPreprocessError` with counts per
      category (never text).

    ``enabled_categories`` restricts which categories count as hits;
    ``None`` enables all categories (defaults plus any ``extra_patterns``).
    """

    mode: Literal["warn", "error", "off"] = "warn"
    enabled_categories: tuple[str, ...] | None = None
    extra_patterns: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("warn", "error", "off"):
            raise TextPreprocessError(f"PHIPolicy.mode must be 'warn', 'error', or 'off'; got {self.mode!r}")
        if self.enabled_categories is not None and any(not c for c in self.enabled_categories):
            raise TextPreprocessError("PHIPolicy.enabled_categories must contain non-empty names")


def check_phi(text: str, policy: PHIPolicy, logger: logging.Logger | None = None) -> list[PHIMatch]:
    """Apply ``policy`` to ``text``; return the hits that survived filtering.

    Logs (warning mode) and raises (error mode) with counts per category
    only — the matched text never appears in any record or message.
    """
    log = logger if logger is not None else _logger
    if policy.mode == "off":
        return []
    matches = phi_scan(text, extra_patterns=policy.extra_patterns)
    if policy.enabled_categories is not None:
        enabled = frozenset(policy.enabled_categories)
        matches = [m for m in matches if m.category in enabled]
    if not matches:
        log.debug("PHI scan clean (0 matches)")
        return matches
    counts = Counter(m.category for m in matches)
    summary = ", ".join(f"{category}={counts[category]}" for category in sorted(counts))
    message = f"PHI screening: {len(matches)} candidate match(es) by category ({summary})"
    if policy.mode == "error":
        raise TextPreprocessError(message)
    log.warning("%s", message)
    return matches
