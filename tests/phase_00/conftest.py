import copy

import pytest

from medfm.tools import governance as gov


@pytest.fixture(scope="session")
def repo_root():
    return gov.REPO_ROOT


@pytest.fixture(scope="session")
def license_schema():
    return gov.load_json(gov.REPO_ROOT / gov.LICENSE_SCHEMA_PATH)


@pytest.fixture(scope="session")
def acceptance_schema():
    return gov.load_json(gov.REPO_ROOT / gov.ACCEPTANCE_SCHEMA_PATH)


@pytest.fixture(scope="session")
def licenses():
    return gov.load_yaml(gov.REPO_ROOT / gov.LICENSES_PATH)


@pytest.fixture(scope="session")
def scope():
    return gov.load_yaml(gov.REPO_ROOT / gov.SCOPE_PATH)


@pytest.fixture()
def valid_license_record(licenses):
    """A known-good record (rad-dino: permissive license, complete fields)."""
    return copy.deepcopy(licenses["rad-dino"])
