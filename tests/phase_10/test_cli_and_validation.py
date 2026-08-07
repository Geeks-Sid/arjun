from __future__ import annotations

import json
import subprocess
import sys


def test_peft_inspect_cli_emits_auditable_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "medfm.cli.peft",
            "inspect",
            "--model",
            "medsiglip_448",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["resolved_model_id"] == "medsiglip_448"
    assert payload["architecture"] == "vision"
    assert payload["selected_count"] > 0
    assert all(
        {"module_name", "module_type", "parameter_count", "selected", "reason"}.issubset(record)
        for record in payload["modules"]
    )


def test_phase_10_contract_validator_has_no_behavioral_errors() -> None:
    from medfm.tools.validate_phase import _check_phase_10_peft

    assert _check_phase_10_peft() == []
