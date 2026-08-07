"""Text and VLM preparation (Phase 04).

Deterministic text canonicalization and instruction-tuning utilities:

- :mod:`medfm.data.textprep.unicode` — Unicode normalization with a fixed,
  documented whitespace/control-character policy.
- :mod:`medfm.data.textprep.phi` — configurable pattern-based PHI screening
  that logs and reports counts per category, never matched text.
- :mod:`medfm.data.textprep.sections` — radiology report section parsing;
  report text is data and is never interpreted as instructions.
- :mod:`medfm.data.textprep.prompts` — prompt templates and conversation
  formatting into :class:`~medfm.core.sample.ConversationTurn` lists.
- :mod:`medfm.data.textprep.tokenize` — supervised-example construction with
  loss masking (only assistant content tokens plus their trailing EOS are
  supervised) and left-side truncation with count-only logging.

Privacy rule (docs/data_governance.md): nothing in this package ever logs or
raises with raw report/conversation text — counts, categories, and positions
only.
"""

from medfm.data.textprep.phi import PHIMatch, PHIPolicy, check_phi, phi_scan
from medfm.data.textprep.prompts import PromptTemplate, PromptTemplateRegistry, format_conversation
from medfm.data.textprep.sections import SECTION_ORDER, ReportSection, parse_report_sections
from medfm.data.textprep.tokenize import (
    SimpleWhitespaceTokenizer,
    SupervisedExample,
    TokenizerProtocol,
    build_supervised_example,
    validate_supervised_batch,
)
from medfm.data.textprep.unicode import normalize_unicode

__all__ = [
    "SECTION_ORDER",
    "PHIMatch",
    "PHIPolicy",
    "PromptTemplate",
    "PromptTemplateRegistry",
    "ReportSection",
    "SimpleWhitespaceTokenizer",
    "SupervisedExample",
    "TokenizerProtocol",
    "build_supervised_example",
    "check_phi",
    "format_conversation",
    "normalize_unicode",
    "parse_report_sections",
    "phi_scan",
    "validate_supervised_batch",
]
