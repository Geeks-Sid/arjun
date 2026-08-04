"""Hardware guards (same policy as tests/phase_01) plus shared fixtures."""

import os

import pytest

GUARDS = {
    "gpu": "MEDFM_RUN_GPU_TESTS",
    "tpu": "MEDFM_RUN_TPU_TESTS",
    "distributed": "MEDFM_RUN_DISTRIBUTED_TESTS",
}


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    for item in items:
        marks = {marker.name for marker in item.iter_markers()}
        for mark, env_var in GUARDS.items():
            if mark in marks and os.environ.get(env_var) != "1":
                item.add_marker(pytest.mark.skip(reason=f"protected {mark} test; set {env_var}=1 to enable"))
