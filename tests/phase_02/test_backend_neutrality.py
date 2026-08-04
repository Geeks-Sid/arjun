"""Backend neutrality: no backend imports or hard-coded devices in core.

Checked structurally via the AST (docstrings may *mention* the policy; code
may not *do* it).
"""

import ast
import inspect

import medfm.core.batch
import medfm.core.encoder
import medfm.core.enums
import medfm.core.errors
import medfm.core.language
import medfm.core.sample
import medfm.core.serialization
import medfm.core.task
import medfm.core.versioning

CORE_MODULES = [
    medfm.core.batch,
    medfm.core.encoder,
    medfm.core.enums,
    medfm.core.errors,
    medfm.core.language,
    medfm.core.sample,
    medfm.core.serialization,
    medfm.core.task,
    medfm.core.versioning,
]

FORBIDDEN_IMPORTS = {"torch_xla", "bitsandbytes", "cucim", "flash_attn"}
FORBIDDEN_METHODS = {"cuda", "xla"}  # .cuda() / .xla() device transfers
FORBIDDEN_DEVICE_PREFIXES = ("cuda", "xla", "tpu", "mps")


def _violations(module) -> list[str]:
    tree = ast.parse(inspect.getsource(module))
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    problems.append(f"import of {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                problems.append(f"import from {node.module} (line {node.lineno})")
        elif isinstance(node, ast.Call):
            # .cuda() / .xla() style hard-coded transfers
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_METHODS:
                problems.append(f"call to .{node.func.attr}() (line {node.lineno})")
            # device="cuda"/"xla"/... constructor arguments
            for keyword in node.keywords:
                if keyword.arg == "device" and isinstance(keyword.value, ast.Constant):
                    value = str(keyword.value.value)
                    if value.split(":")[0] in FORBIDDEN_DEVICE_PREFIXES:
                        problems.append(f"hard-coded device={value!r} (line {node.lineno})")
    return problems


def test_core_modules_have_no_backend_imports_or_hardcoded_devices():
    failures = {module.__name__: problems for module in CORE_MODULES if (problems := _violations(module))}
    assert not failures, f"backend-neutrality violations: {failures}"
