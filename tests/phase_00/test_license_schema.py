"""Schema tests for license records: real registry and synthetic valid/invalid."""

import pytest

from medfm.tools import governance as gov


def test_every_registry_record_is_valid(license_schema, licenses):
    for model_id, record in licenses.items():
        errors = gov.validate_license_record(record, license_schema)
        assert errors == [], f"{model_id}: {errors}"


def test_registry_file_level_validation_passes():
    assert gov.validate_license_file() == {}


def test_valid_record_passes(license_schema, valid_license_record):
    assert gov.validate_license_record(valid_license_record, license_schema) == []


@pytest.mark.parametrize(
    "missing_field",
    [
        "model_id",
        "provider",
        "repository",
        "weights_uri",
        "revision_policy",
        "code_license",
        "weights_license",
        "commercial_use",
        "derivative_models",
        "redistribution",
        "gated_access",
        "accepted_terms_date",
        "approved_use_cases",
        "prohibited_use_cases",
        "status",
        "review_owner",
        "review_date",
        "notes",
    ],
)
def test_missing_required_field_fails(license_schema, valid_license_record, missing_field):
    del valid_license_record[missing_field]
    assert gov.validate_license_record(valid_license_record, license_schema) != []


@pytest.mark.parametrize("field", ["commercial_use", "derivative_models", "redistribution"])
def test_invalid_use_enum_fails(license_schema, valid_license_record, field):
    valid_license_record[field] = "maybe"
    assert gov.validate_license_record(valid_license_record, license_schema) != []


def test_unresolved_terms_must_block(license_schema, valid_license_record):
    """Unresolved terms are blocking rather than guessed."""
    valid_license_record["commercial_use"] = "unresolved"
    valid_license_record["status"] = "pending_review"
    errors = gov.validate_license_record(valid_license_record, license_schema)
    assert any("blocking" in e for e in errors)


def test_unresolved_terms_with_blocked_status_passes(license_schema, valid_license_record):
    valid_license_record["commercial_use"] = "unresolved"
    valid_license_record["status"] = "blocked_unresolved"
    assert gov.validate_license_record(valid_license_record, license_schema) == []


def test_approved_commercial_requires_permitted_use(license_schema, valid_license_record):
    valid_license_record["commercial_use"] = "conditional"
    valid_license_record["status"] = "approved_commercial"
    assert gov.validate_license_record(valid_license_record, license_schema) != []


def test_approved_gated_model_requires_accepted_terms(license_schema, valid_license_record):
    valid_license_record["gated_access"] = True
    valid_license_record["accepted_terms_date"] = None
    valid_license_record["status"] = "approved_research"
    assert gov.validate_license_record(valid_license_record, license_schema) != []


def test_extra_field_fails(license_schema, valid_license_record):
    valid_license_record["surprise"] = "nope"
    assert gov.validate_license_record(valid_license_record, license_schema) != []


def test_key_must_match_model_id(tmp_path, license_schema, valid_license_record):
    import yaml

    valid_license_record["model_id"] = "something-else"
    path = tmp_path / "licenses.yaml"
    path.write_text(yaml.safe_dump({"rad-dino": valid_license_record}))
    problems = gov.validate_license_file(path)
    assert "rad-dino" in problems
