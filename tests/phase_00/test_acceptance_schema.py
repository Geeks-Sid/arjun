"""Schema tests for phase acceptance reports: real phase 00 report and synthetics."""

import copy

import pytest

from medfm.tools import governance as gov


@pytest.fixture()
def valid_report():
    return {
        "phase": "00",
        "status": "passed",
        "generated_by": "test",
        "generated_at": "2026-08-04T00:00:00+00:00",
        "smoke_command": "python -m pytest tests/phase_00 -q",
        "smoke_passed": True,
        "criteria": [
            {
                "id": "c1",
                "description": "something verified",
                "status": "passed",
                "evidence": "pytest output",
            }
        ],
        "unresolved_issues_file": "agent/reports/phase_00/unresolved_issues.md",
        "handoff_file": "agent/reports/phase_00/next_phase_handoff.md",
    }


def test_real_phase_00_acceptance_report_is_valid(acceptance_schema):
    report = gov.load_json(gov.REPO_ROOT / "agent/reports/phase_00/acceptance.json")
    assert gov.validate_acceptance_report(report, acceptance_schema) == []


def test_valid_report_passes(acceptance_schema, valid_report):
    assert gov.validate_acceptance_report(valid_report, acceptance_schema) == []


def test_unknown_criterion_status_rejected(acceptance_schema, valid_report):
    valid_report["criteria"][0]["status"] = "unknown"
    assert gov.validate_acceptance_report(valid_report, acceptance_schema) != []


def test_passed_criterion_requires_evidence(acceptance_schema, valid_report):
    del valid_report["criteria"][0]["evidence"]
    assert gov.validate_acceptance_report(valid_report, acceptance_schema) != []


def test_not_applicable_requires_justification(acceptance_schema, valid_report):
    valid_report["criteria"][0] = {
        "id": "c1",
        "description": "n/a item",
        "status": "not_applicable",
    }
    errors = gov.validate_acceptance_report(valid_report, acceptance_schema)
    assert any("justification" in e for e in errors)


def test_overall_passed_forbids_failed_criteria(acceptance_schema, valid_report):
    valid_report["criteria"][0]["status"] = "failed"
    del valid_report["criteria"][0]["evidence"]
    assert gov.validate_acceptance_report(valid_report, acceptance_schema) != []


def test_overall_passed_requires_smoke_passed(acceptance_schema, valid_report):
    valid_report["smoke_passed"] = False
    assert gov.validate_acceptance_report(valid_report, acceptance_schema) != []


@pytest.mark.parametrize("field", ["phase", "status", "criteria", "smoke_command", "smoke_passed"])
def test_missing_required_field_fails(acceptance_schema, valid_report, field):
    del valid_report[field]
    assert gov.validate_acceptance_report(valid_report, acceptance_schema) != []


def test_bad_phase_format_fails(acceptance_schema, valid_report):
    broken = copy.deepcopy(valid_report)
    broken["phase"] = "0"
    assert gov.validate_acceptance_report(broken, acceptance_schema) != []
