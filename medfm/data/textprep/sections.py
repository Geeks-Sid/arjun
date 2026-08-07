"""Radiology report section parsing.

Recognizes the standard headings INDICATION, HISTORY, COMPARISON, TECHNIQUE,
FINDINGS, and IMPRESSION. A heading is recognized case-insensitively either
alone on its line (optional trailing colon) or at the start of a line
followed by a colon — in the latter case the rest of the line is section
content (``FINDINGS: Lungs are clear.``). Text before the first recognized
heading goes to the PREAMBLE bucket.

The returned mapping always contains every key of :data:`SECTION_ORDER` in
canonical order; sections that are missing or empty map to ``None``.
Repeated headings merge into that section's text.

SECURITY NOTE: report text is DATA. This module never interprets content as
instructions, never evaluates it, and never feeds it to an agent; headings
are recognized by shape only. Callers must apply the same rule downstream
(prompt-injection hygiene for VLM inputs).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Canonical section keys in report order. ``PREAMBLE`` holds any text found
#: before the first recognized heading.
SECTION_ORDER: tuple[str, ...] = (
    "PREAMBLE",
    "INDICATION",
    "HISTORY",
    "COMPARISON",
    "TECHNIQUE",
    "FINDINGS",
    "IMPRESSION",
)

#: Headings recognized as section starts (PREAMBLE is a bucket, not a heading).
_HEADINGS = frozenset(SECTION_ORDER) - {"PREAMBLE"}


@dataclass(frozen=True)
class ReportSection:
    """One parsed report section (name plus verbatim text)."""

    name: str
    text: str


def _heading_split(line: str) -> tuple[str, str] | None:
    """Return ``(heading, inline_content)`` if ``line`` opens a section.

    Recognizes a heading alone on its line (optional trailing colon) and a
    heading followed by a colon plus inline content. A heading-like word
    appearing mid-line is data, not a section start.
    """
    stripped = line.strip()
    if not stripped:
        return None
    if ":" in stripped:
        head, _, rest = stripped.partition(":")
        head = head.strip().upper()
        return (head, rest.strip()) if head in _HEADINGS else None
    return (stripped.upper(), "") if stripped.upper() in _HEADINGS else None


def parse_report_sections(text: str) -> dict[str, ReportSection | None]:
    """Split ``text`` into standard radiology sections.

    Returns an ordered dict over :data:`SECTION_ORDER`; missing and empty
    sections are present with value ``None``. Section text is stripped of
    surrounding blank lines but otherwise verbatim.
    """
    buckets: dict[str, list[str]] = {name: [] for name in SECTION_ORDER}
    current = "PREAMBLE"
    for line in text.split("\n"):
        split = _heading_split(line)
        if split is not None:
            heading, inline = split
            current = heading
            if inline:
                buckets[current].append(inline)
            continue
        buckets[current].append(line)
    sections: dict[str, ReportSection | None] = {}
    for name in SECTION_ORDER:
        body = "\n".join(buckets[name]).strip()
        sections[name] = ReportSection(name=name, text=body) if body else None
    return sections
