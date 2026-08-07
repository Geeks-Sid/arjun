"""Phase 18 release gate: static, CPU-runnable release checks (ADR-gated jobs
plus registry/license/backend-status invariants, backend-neutral import scan,
TPU/NF4 policy scan, clinical-claims scan, and phase-report validation).

CLI: ``python -m medfm.cli.release validate|matrix|checksums``.
``validate`` aggregates every check below and returns nonzero on any failure.
It is designed to run in the Level-1 CI job without GPUs or TPUs; protected
hardware jobs add their own runtime evidence (smoke records) which feed the
registry via ``ModelRegistry.record_backend_result``.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

from medfm.tools import governance as gov

REPO_ROOT = gov.REPO_ROOT

# Registry backend keys each model must resolve (matches BACKEND_KEYS order in
# model_registry/v1_scope.yaml / medfm/registry/schema.py).
BACKEND_KEYS = ("cpu", "cuda_single", "cuda_distributed", "tpu_single_host", "tpu_multi_host")

# A backend status is "resolvable" if it is supported, blocked, or explicitly
# untested/not-applicable — never unknown/ambiguous.
RESOLVABLE_STATUSES = frozenset(
    {
        "SUPPORTED_SINGLE_DEVICE",
        "SUPPORTED_REPLICATED",
        "SUPPORTED_SHARDED",
        "CPU_CONTRACT_ONLY",
        "UNTESTED",
        "BLOCKED_CUSTOM_OP",
        "BLOCKED_MEMORY",
        "BLOCKED_UPSTREAM",
        "NOT_APPLICABLE",
    }
)

# Modules that are backend-hostile when imported eagerly from a
# backend-neutral package (they must only be pulled in lazily/guarded).
HOSTILE_MODULES = ("torch_xla", "bitsandbytes", "cucim", "flash_attn")

# Backend-neutral packages: no *unconditional module-level* imports of the
# hostile modules above (guarded/lazy imports are allowed and expected).
BACKEND_NEUTRAL_PACKAGES = (
    "medfm/core",
    "medfm/data",
    "medfm/tasks",
    "medfm/evaluation",
    "medfm/training",
    "medfm/recipes",
    "medfm/registry",
    "medfm/cli",
)

# Clinical-claim phrases that require a research-status disclaimer in the same
# document. These match claims, not neutral mentions of the words.
CLAIM_PATTERNS = (
    re.compile(r"\bdiagnos(e|es|ed|ing|is)\b", flags=re.IGNORECASE),
    re.compile(r"\bclinically[ -]validated\b", flags=re.IGNORECASE),
    re.compile(r"\bclinical[ -](use|decision|diagnosis)\b", flags=re.IGNORECASE),
    re.compile(r"\bFDA[ -](cleared|approved)\b", flags=re.IGNORECASE),
    re.compile(r"\bsafety[ -]and[ -]efficacy\b", flags=re.IGNORECASE),
)
DISCLAIMER_MARKERS = (
    "research",
    "educational purposes",
    "not for clinical",
    "investigational",
)


# --------------------------------------------------------------------------- #
# Registry / license / backend-status invariants
# --------------------------------------------------------------------------- #


def registry_backend_statuses() -> list[str]:
    """Every catalog model resolves CPU/CUDA/TPU to supported/blocked/untested."""
    errors: list[str] = []
    try:
        from medfm.registry import ModelRegistry, catalog

        catalog.ensure_v1_catalog()
        specs = ModelRegistry.list_models(include_blocked=True, include_deprecated=True)
        if not specs:
            errors.append("catalog exposes no model records")
        for spec in specs:
            missing = set(BACKEND_KEYS) - set(spec.backend_support)
            if missing:
                errors.append(f"{spec.model_id}: backend_support missing keys {sorted(missing)}")
            for key in BACKEND_KEYS:
                support = spec.backend_support.get(key)
                if support is None:
                    errors.append(f"{spec.model_id}: no backend status declared for '{key}'")
                    continue
                status = support.status.value
                if status not in RESOLVABLE_STATUSES:
                    errors.append(
                        f"{spec.model_id}: backend '{key}' has unresolvable status '{status}' "
                        "(must be supported, blocked, or explicitly untested)"
                    )
                if status.startswith("BLOCKED") and not support.blocked_reason:
                    errors.append(f"{spec.model_id}: backend '{key}' BLOCKED without a blocked_reason")
    except Exception as exc:  # noqa: BLE001 - report catalogue failure as a release problem
        errors.append(f"registry backend-status check failed: {type(exc).__name__}: {exc}")
    return errors


def license_registry_consistency() -> list[str]:
    """Reuse the governance license/scope validation (phase-00 contract)."""
    errors: list[str] = []
    for model_id, errs in gov.validate_license_file().items():
        errors.extend(f"license '{model_id}': {e}" for e in errs)
    try:
        scope = gov.load_yaml(REPO_ROOT / gov.SCOPE_PATH)
        license_ids = set(gov.load_yaml(REPO_ROOT / gov.LICENSES_PATH))
        errors.extend(gov.check_scope_consistency(scope, license_ids))
        errors.extend(gov.check_accelerator_policy(scope))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"license/scope consistency check failed: {type(exc).__name__}: {exc}")
    return errors


# --------------------------------------------------------------------------- #
# Static policy scans
# --------------------------------------------------------------------------- #


def no_eager_backend_imports() -> list[str]:
    """Backend-neutral packages must not import CUDA/XLA modules at module top
    level (lazy/guarded imports are fine)."""
    errors: list[str] = []
    for pkg in BACKEND_NEUTRAL_PACKAGES:
        root = REPO_ROOT / Path(*pkg.split("."))
        if not root.is_dir():
            errors.append(f"backend-neutral package missing: {pkg}")
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO_ROOT)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"{rel}: cannot parse source: {exc}")
                continue
            for node in tree.body:
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                roots: set[str] = set()
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
                if isinstance(node, ast.ImportFrom) and node.module:
                    roots.add(node.module.split(".")[0])
                for hostile in HOSTILE_MODULES:
                    if hostile in roots:
                        errors.append(f"{rel}: module-level eager import of '{hostile}' in backend-neutral package")
    return errors


def no_tpu_nf4() -> list[str]:
    """No TPU/XLA config may enable bitsandbytes NF4 quantization."""
    errors: list[str] = []
    config_root = REPO_ROOT / "configs"
    if not config_root.is_dir():
        return errors
    for path in sorted(config_root.rglob("*.yaml")):
        try:
            data = gov.load_yaml(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.relative_to(REPO_ROOT)}: unreadable config: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        backend = str(data.get("accelerator", {}).get("backend", "cpu")).lower()
        if backend not in {"xla_tpu", "tpu"}:
            continue
        peft_block = data.get("peft", {})
        quant = peft_block.get("quantization", {}) if isinstance(peft_block, dict) else {}
        method = str(quant.get("method", "")).lower()
        if method in {"nf4", "bitsandbytes"} or bool(quant.get("bitsandbytes", False)):
            errors.append(f"{path.relative_to(REPO_ROOT)}: TPU config enables bitsandbytes NF4 quantization")
    return errors


def clinical_claims() -> list[str]:
    """Documents carrying clinical-claim language must carry a research-status
    disclaimer (release contains no unsupported clinical claim)."""
    errors: list[str] = []
    scan_roots: list[Path] = [REPO_ROOT / "docs", REPO_ROOT / "README.md"]
    release_dir = REPO_ROOT / "docs" / "release"
    if release_dir.is_dir() and release_dir not in scan_roots:
        scan_roots.append(release_dir)
    files: list[Path] = []
    for root in scan_roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.md")))
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            errors.append(f"{path.relative_to(REPO_ROOT)}: unreadable: {exc}")
            continue
        hits = [pattern.pattern for pattern in CLAIM_PATTERNS if pattern.search(text)]
        if not hits:
            continue
        lowered = text.lower()
        if any(marker in lowered for marker in DISCLAIMER_MARKERS):
            continue
        errors.append(
            f"{path.relative_to(REPO_ROOT)}: clinical-claim language {hits} without a research-status disclaimer"
        )
    return errors


# --------------------------------------------------------------------------- #
# Phase reports and release artifacts
# --------------------------------------------------------------------------- #


def phase_reports() -> list[str]:
    """Every phase acceptance report (00..18) validates and is 'passed'."""
    errors: list[str] = []
    from medfm.tools import validate_phase as vp

    for number in range(0, 19):
        errors.extend(vp.validate_phase(f"{number:02d}"))
    return errors


def validate() -> list[str]:
    """Aggregate all CPU-runnable release checks (no hardware required)."""
    errors: list[str] = []
    errors.extend(registry_backend_statuses())
    errors.extend(license_registry_consistency())
    errors.extend(no_eager_backend_imports())
    errors.extend(no_tpu_nf4())
    errors.extend(clinical_claims())
    errors.extend(phase_reports())
    return errors


# --------------------------------------------------------------------------- #
# Release artifacts: checksums and support matrix
# --------------------------------------------------------------------------- #


def checksums(directory: Path | str) -> dict[str, str]:
    """SHA-256 over every file below ``directory`` (release artifacts)."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"release artifact directory not found: {root}")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def write_checksums(directory: Path | str | None = None, out_path: Path | str | None = None) -> Path:
    """Write ``docs/release/checksums.txt`` for the release artifacts under
    ``directory`` (defaults to ``docs/release``)."""
    root = Path(directory) if directory is not None else REPO_ROOT / "docs" / "release"
    out = Path(out_path) if out_path is not None else REPO_ROOT / "docs" / "release" / "checksums.txt"
    digest = checksums(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{hexdigest}  {relpath}" for relpath, hexdigest in sorted(digest.items())]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def generate_support_matrix(out_path: Path | str | None = None) -> Path:
    """Emit the model x task x backend support matrix from the live catalog."""
    from medfm.registry import ModelRegistry, catalog

    catalog.ensure_v1_catalog()
    specs = ModelRegistry.list_models(include_blocked=True, include_deprecated=True)
    out = Path(out_path) if out_path is not None else REPO_ROOT / "docs" / "release" / "support_matrix.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Release support matrix",
        "",
        "Generated by `python -m medfm.cli.release matrix`. A status of `UNTESTED` is an",
        "explicitly-untested claim (no smoke evidence, not an omission). `BLOCKED_*` requires a",
        "`blocked_reason`; `SUPPORTED_*` requires a recorded smoke revision/date. Protected",
        "hardware jobs upgrade statuses via `ModelRegistry.record_backend_result`.",
        "",
        (
            "| model_id | status | license class | tasks | cpu | cuda_single | "
            "cuda_distributed | tpu_single_host | tpu_multi_host |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows: list[str] = []
    for spec in specs:
        tasks = ",".join(sorted(task.value for task in spec.capabilities.tasks))
        backends = []
        for key in BACKEND_KEYS:
            support = spec.backend_support.get(key)
            backends.append(support.status.value if support is not None else "MISSING")
        rows.append(
            f"| {spec.model_id} | {spec.status.value} | {spec.license.class_type.value} | "
            f"{tasks or '-'} | {' | '.join(backends)} |"
        )
    rows.append("")
    note = [
        "## Supported vs blocked vs untested",
        "",
        "- **supported**: SUPPORTED_SINGLE_DEVICE / SUPPORTED_REPLICATED / SUPPORTED_SHARDED with",
        "  recorded smoke evidence.",
        "- **blocked**: BLOCKED_CUSTOM_OP / BLOCKED_MEMORY / BLOCKED_UPSTREAM with a reason.",
        "- **untested**: UNTESTED (explicit), CPU_CONTRACT_ONLY, or NOT_APPLICABLE.",
        "- Custom third-party CUDA models are **not** TPU-compatible unless declared otherwise.",
        "",
    ]
    out.write_text("\n".join(header + rows + note), encoding="utf-8")
    return out


def report_metrics_claim() -> list[str]:
    """Phase-16 style guard: no evaluation report may claim clinical validation."""
    errors: list[str] = []
    release_dir = REPO_ROOT / "docs" / "release"
    try:
        if release_dir.is_dir():
            for report in sorted(release_dir.rglob("*report*.json")):
                try:
                    data: Any = gov.load_json(report)
                except Exception:  # noqa: BLE001
                    continue
                claims = data.get("claims", {}) if isinstance(data, dict) else {}
                if isinstance(claims, dict) and claims.get("clinically_validated", False) is not False:
                    errors.append(f"{report.relative_to(REPO_ROOT)}: unsupported clinical-validation claim")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"metrics-claim check failed: {type(exc).__name__}: {exc}")
    return errors
