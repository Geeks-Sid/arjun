"""Accelerator policy: per-backend status for every v1 model, and rejection of
blanket cross-accelerator support claims without hardware evidence."""

import copy

from medfm.tools import governance as gov


def test_every_model_has_per_backend_status(scope):
    backend_keys = set(scope["backend_keys"])
    status_enum = set(scope["backend_status_enum"])
    for model in scope["models"]:
        support = model.get("accelerator_support")
        assert isinstance(support, dict), model["model_id"]
        assert set(support) == backend_keys, (
            f"{model['model_id']}: expected backends {sorted(backend_keys)}, got {sorted(support)}"
        )
        for backend, status in support.items():
            assert status in status_enum, f"{model['model_id']}/{backend}: {status}"


def test_registry_passes_accelerator_policy(scope):
    assert gov.check_accelerator_policy(scope) == []


def test_blanket_support_key_is_rejected(scope):
    """A record claiming support via a non-per-backend key (e.g. 'all') fails."""
    broken = copy.deepcopy(scope)
    model = broken["models"][0]
    model["accelerator_support"] = {"all": "SUPPORTED_SINGLE_DEVICE"}
    errors = gov.check_accelerator_policy(broken)
    assert any("blanket" in e or "non-per-backend" in e for e in errors)


def test_supported_status_without_evidence_is_rejected(scope):
    """Claiming SUPPORTED_* without smoke_config + date is rejected."""
    broken = copy.deepcopy(scope)
    model = broken["models"][0]
    model["accelerator_support"]["cuda_single"] = "SUPPORTED_SINGLE_DEVICE"
    model.pop("accelerator_evidence", None)
    errors = gov.check_accelerator_policy(broken)
    assert any("evidence" in e for e in errors)


def test_supported_status_with_evidence_passes(scope):
    fixed = copy.deepcopy(scope)
    model = fixed["models"][0]
    model["accelerator_support"]["cpu"] = "CPU_CONTRACT_ONLY"
    model["accelerator_support"]["cuda_single"] = "SUPPORTED_SINGLE_DEVICE"
    model["accelerator_evidence"] = {
        "cuda_single": {"smoke_config": "configs/smoke/x.yaml", "last_success_date": "2026-08-04"}
    }
    assert gov.check_accelerator_policy(fixed) == []


def test_missing_backend_status_is_rejected(scope):
    broken = copy.deepcopy(scope)
    del broken["models"][0]["accelerator_support"]["tpu_single_host"]
    errors = gov.check_accelerator_policy(broken)
    assert any("missing accelerator status" in e for e in errors)


def test_invalid_status_enum_is_rejected(scope):
    broken = copy.deepcopy(scope)
    broken["models"][0]["accelerator_support"]["cpu"] = "WORKS_GREAT"
    errors = gov.check_accelerator_policy(broken)
    assert any("invalid status" in e for e in errors)
