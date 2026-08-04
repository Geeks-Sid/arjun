"""Phase acceptance gate: python -m medfm.tools.validate_phase --phase <NN>.

Verifies for a completed phase:
  * required files exist,
  * the phase report is populated,
  * acceptance.json validates against agent/acceptance_schema.json,
  * no acceptance criterion is 'unknown',
  * (phase 00) license records and the v1 scope registry are valid and consistent.

Exit code 0 = gate passed, 1 = gate failed, 2 = usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medfm.tools import governance as gov

REPORT_FILES = [
    "summary.md",
    "files_changed.txt",
    "commands_executed.txt",
    "test_results.json",
    "acceptance.json",
    "unresolved_issues.md",
    "next_phase_handoff.md",
]

PHASE_00_REQUIRED_FILES = [
    "docs/product_requirements.md",
    "docs/supported_modalities.md",
    "docs/supported_tasks.md",
    "docs/clinical_safety_scope.md",
    "docs/data_governance.md",
    "docs/model_governance.md",
    "docs/licensing_policy.md",
    "docs/reproducibility_policy.md",
    "docs/architecture/adr_0001_single_framework_multiple_backbones.md",
    "docs/architecture/adr_0002_peft_first_training.md",
    "docs/architecture/adr_0003_external_encoder_vlm_bridge.md",
    "docs/architecture/adr_0004_patient_level_splitting.md",
    "docs/architecture/adr_0005_native_3d_and_slice_sequence_vlm.md",
    "docs/architecture/adr_0006_adapter_only_checkpoints.md",
    "docs/architecture/adr_0007_pytorch_cuda_and_xla_backends.md",
    "docs/architecture/adr_0008_tpu_static_shape_buckets.md",
    "docs/architecture/adr_0009_cuda_qlora_vs_tpu_bf16_lora.md",
    "model_registry/license_schema.json",
    "model_registry/licenses.yaml",
    "model_registry/v1_scope.yaml",
    "agent/README.md",
    "agent/phase_template.md",
    "agent/acceptance_schema.json",
    "agent/prompts/implement_phase.md",
    "agent/prompts/review_phase.md",
    "agent/prompts/test_phase.md",
    "agent/prompts/repair_phase.md",
]


def _check_report(phase: str, errors: list[str]) -> None:
    report_dir = gov.REPO_ROOT / "agent" / "reports" / f"phase_{phase}"
    for name in REPORT_FILES:
        path = report_dir / name
        if not path.exists():
            errors.append(f"missing phase report file: {path.relative_to(gov.REPO_ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"empty phase report file: {path.relative_to(gov.REPO_ROOT)}")

    acceptance_path = report_dir / "acceptance.json"
    if acceptance_path.exists() and acceptance_path.stat().st_size > 0:
        report = gov.load_json(acceptance_path)
        schema = gov.load_json(gov.REPO_ROOT / gov.ACCEPTANCE_SCHEMA_PATH)
        errors.extend(validate_acceptance(report, schema, phase))

    test_results = report_dir / "test_results.json"
    if test_results.exists() and test_results.stat().st_size > 0:
        data = gov.load_json(test_results)
        if not data.get("tests"):
            errors.append("test_results.json contains no tests")
        for t in data.get("tests", []):
            if t.get("status") not in {"passed", "failed", "skipped"}:
                errors.append(f"test_results.json: test '{t.get('name')}' has invalid status")
            if t.get("status") == "skipped" and not t.get("reason"):
                errors.append(f"test_results.json: skipped test '{t.get('name')}' lacks a reason")


def validate_acceptance(report: dict, schema: dict, phase: str) -> list[str]:
    errors = gov.validate_acceptance_report(report, schema)
    if errors:
        return errors
    if report["phase"] != phase:
        errors.append(f"acceptance.json phase '{report['phase']}' != requested '{phase}'")
    if report["status"] != "passed":
        errors.append(f"acceptance.json status is '{report['status']}', gate requires 'passed'")
    return errors


def validate_phase(phase: str) -> list[str]:
    errors: list[str] = []
    if phase == "00":
        for rel in PHASE_00_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        for model_id, errs in gov.validate_license_file().items():
            errors.extend(f"license '{model_id}': {e}" for e in errs)
        scope = gov.load_yaml(gov.REPO_ROOT / gov.SCOPE_PATH)
        license_ids = set(gov.load_yaml(gov.REPO_ROOT / gov.LICENSES_PATH))
        errors.extend(gov.check_scope_consistency(scope, license_ids))
        errors.extend(gov.check_accelerator_policy(scope))
    else:
        errors.append(f"no validator registered for phase {phase}")
    _check_report(phase, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a phase acceptance gate.")
    parser.add_argument("--phase", required=True, help="phase number, e.g. 00")
    args = parser.parse_args(argv)
    phase = args.phase.zfill(2)

    errors = validate_phase(phase)
    if errors:
        print(f"Phase {phase} gate FAILED ({len(errors)} problem(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"Phase {phase} gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
