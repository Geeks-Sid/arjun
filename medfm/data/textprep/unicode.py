"""Unicode normalization for clinical text.

Policy (fixed and deterministic — the same input always yields the same
output, so downstream cache keys and fingerprints are stable):

- **Normalization form**: NFC by default (canonical composition, the form
  most tokenizers expect). NFKC is available for compatibility folding
  (e.g. full-width digits, ligatures); NFKD/NFD are deliberately not
  accepted because decomposed forms change token boundaries for no benefit
  in this pipeline.
- **Line endings**: ``\\r\\n`` and ``\\r`` are folded to ``\\n`` BEFORE
  control-character stripping so line structure survives.
- **Control characters**: Unicode category ``Cc`` is stripped except TAB and
  LF (meaningful layout in reports). Category ``Cf`` (zero-width characters,
  bidi controls, soft hyphens) is stripped entirely — invisible format
  characters carry no clinical signal and can hide prompt-injection content.
- **Whitespace**: runs of horizontal whitespace (space/tab) collapse to one
  space; trailing whitespace is removed per line; runs of 3+ newlines
  collapse to one blank line (``\\n\\n``); the whole string is stripped.
"""

from __future__ import annotations

import unicodedata
from typing import Literal, cast

from medfm.data.errors import TextPreprocessError

#: Normalization forms this module accepts (see module docstring).
_ALLOWED_FORMS = ("NFC", "NFKC")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Return ``text`` normalized per the deterministic policy above.

    Raises :class:`TextPreprocessError` if ``form`` is not NFC or NFKC.
    Never logs or raises with the input text (privacy rule).
    """
    if form not in _ALLOWED_FORMS:
        raise TextPreprocessError(f"normalize_unicode form must be one of {_ALLOWED_FORMS}; got {form!r}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize(cast("Literal['NFC', 'NFKC']", form), text)
    kept = [ch for ch in text if ch in ("\t", "\n") or (unicodedata.category(ch) not in ("Cc", "Cf"))]
    lines = [_collapse_horizontal(line) for line in "".join(kept).split("\n")]
    collapsed = "\n".join(lines)
    while "\n\n\n" in collapsed:
        collapsed = collapsed.replace("\n\n\n", "\n\n")
    return collapsed.strip()


def _collapse_horizontal(line: str) -> str:
    """Collapse runs of spaces/tabs to one space and strip the line's ends."""
    out: list[str] = []
    pending_space = False
    for ch in line:
        if ch in (" ", "\t"):
            pending_space = True
        else:
            if pending_space and out:
                out.append(" ")
            out.append(ch)
            pending_space = False
    return "".join(out)
