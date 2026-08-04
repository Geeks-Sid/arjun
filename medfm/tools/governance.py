"""Governance validation helpers shared by tests and the phase validator.

Phase 00 tooling: validates license records, phase acceptance reports, and the
machine-readable v1 scope registry. No model, data, or training code lives here.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]

LICENSES_PATH = Path("model_registry/licenses.yaml")
LICENSE_SCHEMA_PATH = Path("model_registry/license_schema.json")
SCOPE_PATH = Path("model_registry/v1_scope.yaml")
ACCEPTANCE_SCHEMA_PATH = Path("agent/acceptance_schema.json")

_LICENSE_USE_VALUES = {"permitted", "prohibited", "conditional", "unresolved"}
_LICENSE_STATUSES = {
    "approved_research",
    "approved_commercial",
    "pending_review",
    "blocked_unresolved",
    "rejected",
}
_APPROVED_STATUSES = {"approved_research", "approved_commercial"}
_SUPPORTED_BACKEND_STATUSES = {
    "SUPPORTED_SINGLE_DEVICE",
    "SUPPORTED_REPLICATED",
    "SUPPORTED_SHARDED",
}


def load_yaml(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _schema_errors(instance: Any, schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return sorted(e.message for e in validator.iter_errors(instance))


def _normalize_dates(value: Any) -> Any:
    """YAML parses ISO dates into datetime.date; schemas expect strings."""
    if isinstance(value, dict):
        return {k: _normalize_dates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_dates(v) for v in value]
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    return value


def validate_license_record(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate one license record against the schema plus cross-field policy."""
    record = _normalize_dates(record)
    errors = _schema_errors(record, schema)
    if errors:
        return errors  # cross-field rules need well-typed fields

    use_fields = [record["commercial_use"], record["derivative_models"], record["redistribution"]]
    status = record["status"]

    if any(v == "unresolved" for v in use_fields) and status != "blocked_unresolved":
        errors.append("unresolved license terms are blocking: status must be 'blocked_unresolved'")
    if (record["repository"] == "unresolved" or record["weights_uri"] == "unresolved") and status not in {
        "blocked_unresolved",
        "pending_review",
    }:
        errors.append("unresolved source/weights location forbids an approved status")
    if status in _APPROVED_STATUSES:
        if record["gated_access"] and not record["accepted_terms_date"]:
            errors.append("approved gated model requires accepted_terms_date")
        if not record["review_owner"] or not record["review_date"]:
            errors.append("approved model requires review_owner and review_date")
    if status == "approved_commercial" and record["commercial_use"] != "permitted":
        errors.append("approved_commercial requires commercial_use == 'permitted'")
    return errors


def validate_license_file(path: Path | str = LICENSES_PATH) -> dict[str, list[str]]:
    """Validate every record in a licenses file. Returns model_id -> errors."""
    schema = load_json(REPO_ROOT / LICENSE_SCHEMA_PATH)
    data = load_yaml(REPO_ROOT / path) if not Path(path).is_absolute() else load_yaml(path)
    if not isinstance(data, dict) or not data:
        return {"<file>": ["licenses file must be a non-empty mapping of model_id to record"]}
    problems: dict[str, list[str]] = {}
    for key, record in data.items():
        if isinstance(record, dict):
            errs = validate_license_record(record, schema)
        else:
            errs = _schema_errors(record, schema)
        if isinstance(record, dict) and record.get("model_id") != key:
            errs = errs + [f"mapping key '{key}' does not match model_id '{record.get('model_id')}'"]
        if errs:
            problems[key] = errs
    return problems


def validate_acceptance_report(report: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate a phase acceptance report. 'unknown' is never a legal status."""
    errors = _schema_errors(report, schema)
    if errors:
        return errors
    for i, criterion in enumerate(report["criteria"]):
        status = criterion["status"]
        if status == "unknown":
            errors.append(f"criterion[{i}] '{criterion['id']}': 'unknown' status is not allowed")
        if status == "passed" and not criterion.get("evidence"):
            errors.append(f"criterion[{i}] '{criterion['id']}': passed requires evidence")
        if status in {"blocked", "not_applicable"} and not criterion.get("justification"):
            errors.append(f"criterion[{i}] '{criterion['id']}': {status} requires justification")
    if report["status"] == "passed" and not report["smoke_passed"]:
        errors.append("overall status 'passed' requires smoke_passed == true")
    if report["status"] == "passed" and any(c["status"] in {"failed", "blocked"} for c in report["criteria"]):
        errors.append("overall status 'passed' forbids failed/blocked criteria")
    return errors


def check_scope_consistency(scope: dict[str, Any], license_ids: set[str]) -> list[str]:
    """Cross-checks between the v1 scope registry and the license records."""
    errors: list[str] = []

    modalities = {m["name"] for m in scope["modalities"]}
    tasks = {t["name"] for t in scope["tasks"]}
    models = {m["model_id"]: m for m in scope["models"]}

    # Every scope model (v1 or deferred) has a license record; no orphan records.
    for model_id in models:
        if model_id not in license_ids:
            errors.append(f"model '{model_id}' has no license record")
    for lic_id in license_ids - set(models):
        errors.append(f"license record '{lic_id}' has no scope entry")

    # Every modality has at least one backbone candidate that exists as a model.
    for mod in scope["modalities"]:
        candidates = mod.get("backbone_candidates") or []
        if not candidates:
            errors.append(f"modality '{mod['name']}' has no backbone candidate")
        for cand in candidates:
            if cand not in models:
                errors.append(f"modality '{mod['name']}' candidate '{cand}' is not a registered model")
        for role_key in ("preferred_backbone", "fallback_backbone"):
            role_val = mod.get(role_key)
            if role_val and role_val not in candidates:
                errors.append(f"modality '{mod['name']}' {role_key} '{role_val}' is not in backbone_candidates")

    # Matrix is a complete partition of modalities x tasks.
    matrix = scope["modality_task_matrix"]
    for mod_name in modalities:
        if mod_name not in matrix:
            errors.append(f"matrix missing modality '{mod_name}'")
            continue
        row = matrix[mod_name]
        seen: dict[str, str] = {}
        for bucket in ("supported", "deferred", "unsupported"):
            for task in row.get(bucket) or []:
                if task not in tasks:
                    errors.append(f"matrix '{mod_name}' lists unknown task '{task}'")
                if task in seen:
                    errors.append(f"matrix '{mod_name}' task '{task}' in both '{seen[task]}' and '{bucket}'")
                seen[task] = bucket
        missing = tasks - set(seen)
        if missing:
            errors.append(f"matrix '{mod_name}' has no disposition for tasks: {sorted(missing)}")
    for mod_name in set(matrix) - modalities:
        errors.append(f"matrix lists unknown modality '{mod_name}'")

    # Every task supported for at least one modality has an implementation path
    # and at least one model that lists the task.
    task_models: dict[str, set[str]] = {t: set() for t in tasks}
    for model in models.values():
        for t in model.get("tasks") or []:
            if t not in tasks:
                errors.append(f"model '{model['model_id']}' lists unknown task '{t}'")
            else:
                task_models[t].add(model["model_id"])
        for m in model.get("modalities") or []:
            if m not in modalities:
                errors.append(f"model '{model['model_id']}' lists unknown modality '{m}'")
    for task in scope["tasks"]:
        if not task.get("implementation_path"):
            errors.append(f"task '{task['name']}' has no implementation path")
        supported_somewhere = any(task["name"] in (matrix[m].get("supported") or []) for m in matrix)
        if supported_somewhere:
            claimed = set(task.get("primary_models") or [])
            if not claimed:
                errors.append(f"task '{task['name']}' is supported but has no primary model")
            for pm in claimed:
                if pm not in models:
                    errors.append(f"task '{task['name']}' primary model '{pm}' is not registered")
                elif task["name"] not in (models[pm].get("tasks") or []):
                    errors.append(
                        f"task '{task['name']}' claims primary model '{pm}' but the model does not list the task"
                    )
            if not task_models[task["name"]]:
                errors.append(f"task '{task['name']}' is supported but no model implements it")

    # Vertical slices reference registered modalities/tasks/models.
    for vs in scope.get("vertical_slices") or []:
        if vs["modality"] not in modalities:
            errors.append(f"vertical slice '{vs['id']}' uses unknown modality '{vs['modality']}'")
        if vs["task"] not in tasks:
            errors.append(f"vertical slice '{vs['id']}' uses unknown task '{vs['task']}'")
        for model_id in vs.get("models") or []:
            if model_id not in models:
                errors.append(f"vertical slice '{vs['id']}' uses unknown model '{model_id}'")

    return errors


def check_accelerator_policy(scope: dict[str, Any]) -> list[str]:
    """Every model records a per-backend status; no blanket support claims."""
    errors: list[str] = []
    backend_keys = set(scope["backend_keys"])
    status_enum = set(scope["backend_status_enum"])

    for model in scope["models"]:
        model_id = model["model_id"]
        support = model.get("accelerator_support")
        if not isinstance(support, dict):
            errors.append(f"model '{model_id}' has no accelerator_support mapping")
            continue
        illegal_keys = set(support) - backend_keys
        if illegal_keys:
            errors.append(
                f"model '{model_id}' uses non-per-backend support keys {sorted(illegal_keys)} "
                "(blanket cross-accelerator claims are not allowed)"
            )
        missing = backend_keys - set(support)
        if missing:
            errors.append(f"model '{model_id}' missing accelerator status for {sorted(missing)}")
        evidence = model.get("accelerator_evidence") or {}
        for backend, status in support.items():
            if backend not in backend_keys:
                continue
            if status not in status_enum:
                errors.append(f"model '{model_id}' backend '{backend}' has invalid status '{status}'")
                continue
            if status in _SUPPORTED_BACKEND_STATUSES:
                ev = evidence.get(backend)
                if not ev or not ev.get("smoke_config") or not ev.get("last_success_date"):
                    errors.append(
                        f"model '{model_id}' claims '{status}' on '{backend}' without recorded "
                        "hardware evidence (smoke_config + last_success_date)"
                    )
    return errors
