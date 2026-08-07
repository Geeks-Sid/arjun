"""Unicode normalization, PHI screening, section parsing, and prompt templates."""

import logging

import pytest

from medfm.data.errors import TextPreprocessError
from medfm.data.textprep import (
    SECTION_ORDER,
    PHIMatch,
    PHIPolicy,
    PromptTemplate,
    PromptTemplateRegistry,
    ReportSection,
    check_phi,
    format_conversation,
    normalize_unicode,
    parse_report_sections,
    phi_scan,
)

# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------


def test_nfc_composes_accents():
    # "e" + combining acute must become the single precomposed code point.
    assert normalize_unicode("café") == "café"


def test_nfkc_folds_compatibility_characters():
    assert normalize_unicode("①②", form="NFKC") == "12"
    # NFC leaves compatibility characters untouched.
    assert normalize_unicode("①②", form="NFC") == "①②"


def test_disallowed_unicode_form_rejected():
    with pytest.raises(TextPreprocessError):
        normalize_unicode("x", form="NFD")


def test_control_characters_stripped_but_layout_kept():
    text = "line one\x00\x07\ttwo\x7f\u0085three"
    out = normalize_unicode(text)
    for ch in out:
        assert ch in ("\n", "\t") or ord(ch) >= 0x20
    assert "one" in out and "three" in out


def test_zero_width_and_bidi_format_chars_stripped():
    assert normalize_unicode("ab\u200bc\ufeffd\u202ee") == "abcde"


def test_line_endings_folded_to_lf():
    assert normalize_unicode("a\r\nb\rc") == "a\nb\nc"


def test_whitespace_policy_is_deterministic():
    raw = "  hello   \t world  \n\n\n\ntrailing   \n"
    out = normalize_unicode(raw)
    assert out == "hello world\n\ntrailing"
    assert normalize_unicode(out) == out  # idempotent


# ---------------------------------------------------------------------------
# PHI screening
# ---------------------------------------------------------------------------

PHI_TEXT = (
    "Patient MRN: 1234567 was seen on 01/15/1980. Call back at 555-123-4567 or email jdoe@example.org. SSN 123-45-6789."
)
RAW_PHI_FRAGMENTS = ("1234567", "01/15/1980", "555-123-4567", "jdoe@example.org", "123-45-6789")


def test_phi_scan_finds_mrn_date_phone_email_ssn():
    matches = phi_scan(PHI_TEXT)
    categories = {m.category for m in matches}
    assert {"mrn", "date", "phone", "email", "ssn"} <= categories
    assert all(isinstance(m, PHIMatch) for m in matches)


def test_phi_match_never_carries_text():
    for match in phi_scan(PHI_TEXT):
        assert set(match.__dict__) == {"category", "start", "end"}
        assert 0 <= match.start <= match.end


def test_phi_scan_extra_patterns_extend_and_override():
    extra = {"accession": r"\bACC\d{6}\b", "ssn": r"\bSSN:\d{9}\b"}
    matches = phi_scan("ACC123456 noted; SSN:123456789 on file.", extra_patterns=extra)
    categories = [m.category for m in matches]
    assert "accession" in categories
    assert categories.count("ssn") == 1  # caller pattern replaced the default


def test_phi_scan_invalid_extra_regex_raises():
    with pytest.raises(TextPreprocessError):
        phi_scan("text", extra_patterns={"broken": r"([unclosed"})


def test_phi_policy_off_skips_scanning():
    assert check_phi(PHI_TEXT, PHIPolicy(mode="off")) == []


def test_phi_policy_warn_logs_counts_only(caplog):
    with caplog.at_level(logging.WARNING, logger="medfm.data.textprep.phi"):
        matches = check_phi(PHI_TEXT, PHIPolicy(mode="warn"))
    assert matches
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "ssn=1" in message and "phone=1" in message and "email=1" in message
    for fragment in RAW_PHI_FRAGMENTS:
        assert fragment not in message
        assert fragment not in caplog.text


def test_phi_policy_error_raises_counts_only():
    with pytest.raises(TextPreprocessError) as excinfo:
        check_phi(PHI_TEXT, PHIPolicy(mode="error"))
    message = str(excinfo.value)
    assert "candidate match(es)" in message
    for fragment in RAW_PHI_FRAGMENTS:
        assert fragment not in message


def test_phi_policy_enabled_categories_filters():
    matches = check_phi(PHI_TEXT, PHIPolicy(mode="warn", enabled_categories=("ssn",)))
    assert {m.category for m in matches} == {"ssn"}


def test_phi_policy_clean_text_logs_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="medfm.data.textprep.phi"):
        assert check_phi("No acute cardiopulmonary abnormality.", PHIPolicy(mode="warn")) == []
    assert caplog.records == []


def test_phi_policy_invalid_mode_rejected():
    with pytest.raises(TextPreprocessError):
        PHIPolicy(mode="redact")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Report section parsing
# ---------------------------------------------------------------------------

FULL_REPORT = """Preliminary note from technologist.
INDICATION: Chest pain.
HISTORY:
Smoker, 40 pack-years.
COMPARISON: None available.
TECHNIQUE: PA and lateral chest radiographs.
FINDINGS:
Lungs are clear.
No pleural effusion.
IMPRESSION:
No acute cardiopulmonary abnormality.
"""


def test_parse_full_report_all_sections_present():
    sections = parse_report_sections(FULL_REPORT)
    assert list(sections) == list(SECTION_ORDER)
    assert sections["PREAMBLE"].text == "Preliminary note from technologist."
    assert sections["INDICATION"].text == "Chest pain."
    assert "40 pack-years" in sections["HISTORY"].text
    assert sections["FINDINGS"].text == "Lungs are clear.\nNo pleural effusion."
    assert "No acute" in sections["IMPRESSION"].text
    assert all(isinstance(s, ReportSection) for s in sections.values())


def test_parse_headings_are_case_insensitive():
    sections = parse_report_sections("findings:\nClear.\nimpression: Unremarkable.")
    assert sections["FINDINGS"].text == "Clear."
    assert sections["IMPRESSION"].text == "Unremarkable."
    assert sections["PREAMBLE"] is None


def test_parse_missing_sections_are_none():
    sections = parse_report_sections("FINDINGS: Lungs are clear.")
    assert sections["FINDINGS"].text == "Lungs are clear."
    for name in ("PREAMBLE", "INDICATION", "HISTORY", "COMPARISON", "TECHNIQUE", "IMPRESSION"):
        assert sections[name] is None


def test_parse_empty_section_is_none():
    sections = parse_report_sections("FINDINGS:\nIMPRESSION: Stable.")
    assert sections["FINDINGS"] is None
    assert sections["IMPRESSION"].text == "Stable."


def test_parse_preamble_bucket_collects_text_before_first_heading():
    sections = parse_report_sections("Dictated but not read.\nSecond preamble line.\nIMPRESSION: Normal.")
    assert sections["PREAMBLE"].text == "Dictated but not read.\nSecond preamble line."
    assert sections["IMPRESSION"].text == "Normal."


def test_parse_repeated_heading_merges_text():
    sections = parse_report_sections("FINDINGS: First.\nIMPRESSION: X.\nFINDINGS: Addendum line.")
    assert sections["FINDINGS"].text == "First.\nAddendum line."


def test_parse_text_looking_like_heading_mid_line_is_data():
    sections = parse_report_sections("FINDINGS: The word IMPRESSION: appears inline.")
    assert sections["IMPRESSION"] is None
    assert "IMPRESSION:" in sections["FINDINGS"].text


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def _vqa_template() -> PromptTemplate:
    return PromptTemplate(
        template_id="vqa.chest.v1",
        system="You are a radiology assistant.",
        user_template="Question: {question}",
        version="1",
    )


def test_registry_register_and_get():
    registry = PromptTemplateRegistry()
    template = _vqa_template()
    registry.register(template)
    assert registry.get("vqa.chest.v1") is template


def test_registry_duplicate_id_rejected():
    registry = PromptTemplateRegistry()
    registry.register(_vqa_template())
    with pytest.raises(TextPreprocessError):
        registry.register(_vqa_template())


def test_registry_unknown_id_rejected():
    with pytest.raises(TextPreprocessError):
        PromptTemplateRegistry().get("nope")


def test_format_conversation_builds_system_user_assistant_turns():
    turns = format_conversation(_vqa_template(), {"question": "Is the chest xray normal?"}, "Yes, it is normal.")
    assert [t.role for t in turns] == ["system", "user", "assistant"]
    assert turns[0].content == "You are a radiology assistant."
    assert turns[1].content == "Question: Is the chest xray normal?"
    assert turns[2].content == "Yes, it is normal."


def test_format_conversation_without_system_or_assistant():
    template = PromptTemplate(template_id="t", user_template="Report findings: {findings}", version="2")
    turns = format_conversation(template, {"findings": "Lungs are clear."})
    assert [t.role for t in turns] == ["user"]
    assert turns[0].content == "Report findings: Lungs are clear."


def test_format_conversation_extra_fields_ignored():
    turns = format_conversation(_vqa_template(), {"question": "Q?", "unused": "x"}, "A.")
    assert turns[1].content == "Question: Q?"


def test_format_conversation_unknown_placeholder_rejected():
    with pytest.raises(TextPreprocessError, match="unknown placeholder"):
        format_conversation(_vqa_template(), {"findings": "x"}, "A.")


def test_template_validation_rejects_empty_fields():
    with pytest.raises(TextPreprocessError):
        PromptTemplate(template_id="", user_template="{question}")
    with pytest.raises(TextPreprocessError):
        PromptTemplate(template_id="t", user_template="")
