"""Phase 18 golden regression API (Level 4, CPU).

Every pinned fixture is re-derived from the code under test and compared within
a dtype-tolerant bound. A committed SHA-256 manifest detects accidental edits,
so upstream drift (torch / torchvision / MONAI) that changes shapes, preprocess
statistics, logits, masks, structured fields, or memory envelopes fails here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys

import pytest

from medfm.tools import governance as gov

GOLDEN_DIR = gov.REPO_ROOT / "tests" / "phase_18" / "golden"


def _golden_generator():
    path = gov.REPO_ROOT / "scripts" / "generate_golden.py"
    spec = importlib.util.spec_from_file_location("medfm_golden_gen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GK = _golden_generator()
FIXTURE_NAMES = sorted(GK.FIXTURES)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_golden_fixture_matches_pinned_value(name: str) -> None:
    golden = json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))
    produced = GK.FIXTURES[name]()
    _assert_close(produced, golden, name)


def test_golden_manifest_hashes_are_intact() -> None:
    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] >= 1
    assert set(manifest["files"]) == set(FIXTURE_NAMES)
    for name, digest in manifest["files"].items():
        actual = hashlib.sha256((GOLDEN_DIR / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} was hand-edited; regenerate with scripts/generate_golden.py"


def test_memory_envelope_holds_parameter_footprint() -> None:
    envelope = json.loads((GOLDEN_DIR / "memory_envelope.json").read_text(encoding="utf-8"))
    produced = GK.memory_envelope()
    assert produced["parameter_bytes"] == envelope["parameter_bytes"]
    assert produced["parameter_bytes"] <= envelope["envelope_bytes"]


def _assert_close(actual: object, expected: object, name: str) -> None:
    if isinstance(expected, float):
        assert isinstance(actual, (int, float))
        assert float(actual) == pytest.approx(expected, rel=1e-5, abs=1e-6), name
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, list) and isinstance(expected, list) or isinstance(actual, tuple)
        assert len(actual) == len(expected), name
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):  # type: ignore[arg-type]
            _assert_close(actual_item, expected_item, f"{name}[{index}]")
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), name
        assert set(actual) == set(expected), name
        for key, value in expected.items():
            _assert_close(actual[key], value, f"{name}.{key}")  # type: ignore[index]
    else:
        assert actual == expected, name
