"""Repository-wide protected-hardware / protected-weight test guards (Phase 18).

Level 3 real-checkpoint and other gated tests must not silently pass on a
machine without the required weights or hardware: they are explicitly skipped
unless the opt-in environment variable is set, and the CI protected jobs set it
before running. The per-phase ``conftest.py`` files apply the same policy to the
``gpu``/``tpu``/``distributed`` markers.
"""

from __future__ import annotations

import os

import pytest

PROTECTED_MARKS: dict[str, str] = {
    # Level 3: real model checkpoints (MedSigLIP, 3D CT, MRI/task, pathology,
    # MedGemma). Never runs in ordinary CPU jobs.
    "real_checkpoint": "MEDFM_RUN_REAL_CHECKPOINTS",
}


def pytest_collection_modifyitems(config, items) -> None:  # noqa: ARG001
    for item in items:
        marks = {marker.name for marker in item.iter_markers()}
        for mark, env_var in PROTECTED_MARKS.items():
            if mark in marks and os.environ.get(env_var) != "1":
                item.add_marker(pytest.mark.skip(reason=f"protected {mark} test; set {env_var}=1 to enable"))
