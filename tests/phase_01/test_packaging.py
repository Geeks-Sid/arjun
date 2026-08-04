"""Package layout and import safety on CPU-only processes."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SUBPACKAGES = [
    "cli",
    "core",
    "data",
    "models",
    "peft",
    "tasks",
    "training",
    "evaluation",
    "inference",
    "registry",
    "tools",
]

#: CUDA-only or XLA-only libraries that must never be imported at module
#: top level by first-party code (lazy, function-local imports are allowed).
FORBIDDEN_TOP_LEVEL_IMPORTS = ("bitsandbytes", "torch_xla", "flash_attn", "cucim")


def test_subpackages_importable():
    import importlib

    for name in SUBPACKAGES:
        importlib.import_module(f"medfm.{name}")


def _first_party_modules() -> list[Path]:
    return sorted((REPO_ROOT / "medfm").rglob("*.py"))


def test_no_forbidden_top_level_imports():
    offenders: list[str] = []
    for path in _first_party_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # top-level statements only
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_TOP_LEVEL_IMPORTS:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")
    assert not offenders, f"forbidden top-level imports: {offenders}"


@pytest.mark.skipif(not (REPO_ROOT / "uv.lock").exists(), reason="uv.lock not generated")
def test_cpu_import_does_not_initialize_cuda_or_xla():
    code = (
        "import sys; "
        "import medfm, medfm.cli.main, medfm.tools.doctor, medfm.tools.smoke, "
        "medfm.training.run_metadata, medfm.training.tracking; "
        "import torch; "
        "assert not torch.cuda.is_initialized(), 'CUDA initialized on import'; "
        "assert 'torch_xla' not in sys.modules, 'torch_xla imported'; "
        "assert 'bitsandbytes' not in sys.modules, 'bitsandbytes imported'"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}  # CUDA present but must stay untouched
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
