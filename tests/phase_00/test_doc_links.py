"""Internal link/reference check for policy documents, and ownership metadata."""

import re
from pathlib import Path

import pytest

from medfm.tools import governance as gov

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

DOC_DIRS = ["docs", "agent", "model_registry"]

GOVERNANCE_DOCS = [
    "docs/clinical_safety_scope.md",
    "docs/data_governance.md",
    "docs/model_governance.md",
    "docs/licensing_policy.md",
    "docs/reproducibility_policy.md",
]


def _markdown_files():
    files = []
    for d in DOC_DIRS:
        files.extend((gov.REPO_ROOT / d).rglob("*.md"))
    return files


def _iter_internal_links(path: Path):
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for target in LINK_RE.findall(line):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            yield lineno, target.split("#", 1)[0]


def test_internal_links_resolve():
    broken = []
    for md in _markdown_files():
        for lineno, target in _iter_internal_links(md):
            if not target:  # pure anchor after stripping
                continue
            if not (md.parent / target).resolve().exists():
                broken.append(f"{md.relative_to(gov.REPO_ROOT)}:{lineno} -> {target}")
    assert broken == [], "broken internal links:\n" + "\n".join(broken)


@pytest.mark.parametrize("doc", GOVERNANCE_DOCS)
def test_governance_doc_has_owner_and_review_date(doc):
    text = (gov.REPO_ROOT / doc).read_text(encoding="utf-8")
    assert re.search(r"^Owner: \S", text, re.MULTILINE), f"{doc} lacks an owner"
    assert re.search(r"^Review date: \d{4}-\d{2}-\d{2}", text, re.MULTILINE), f"{doc} lacks a review date"


def test_every_adr_has_required_sections():
    adrs = sorted((gov.REPO_ROOT / "docs/architecture").glob("adr_*.md"))
    assert len(adrs) == 9, f"expected 9 ADRs, found {len(adrs)}"
    for adr in adrs:
        text = adr.read_text(encoding="utf-8")
        for section in (
            "## Context",
            "## Decision",
            "## Alternatives considered",
            "## Consequences",
            "## Reversal conditions",
        ):
            assert section in text, f"{adr.name} missing '{section}'"
