"""Runtime diagnostics: ``medfm doctor`` / ``python -m medfm.tools.doctor``.

Reports Python, PyTorch, CUDA/driver/GPU/VRAM, BF16 and SDPA/FlashAttention
support, key package versions, disk/cache status, and — when explicitly
selected — TPU runtime details (PyTorch-XLA, libtpu, topology, SPMD).

Import discipline: ``torch_xla`` is only imported when the user explicitly
selects the ``xla_tpu`` backend, and CUDA is never initialized for a CPU
report. No credentials, tokens, or patient-data paths are printed; cache
paths are displayed with the home directory masked.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Packages whose presence in a TPU environment is a red flag (CUDA-only).
TPU_INCOMPATIBLE_PACKAGES = ("bitsandbytes", "flash-attn", "cucim-cu12", "cucim")

#: Environment-variable name fragments that must never be printed.
SENSITIVE_ENV_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _mask_home(path: Path) -> str:
    """Mask the user's home directory so no absolute user paths leak."""
    try:
        return "~" + os.sep + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _cache_status(env_var: str, default: Path) -> dict[str, Any]:
    raw = os.environ.get(env_var)
    path = Path(raw).expanduser() if raw else default
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return {
        "path": _mask_home(path),
        "configured": raw is not None,
        "exists": path.exists(),
        "writable": os.access(probe, os.W_OK),
    }


def _python_section() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": _mask_home(Path(sys.executable)),
    }


def _packages_section() -> dict[str, Any]:
    # Version lookups only — importing these packages must never happen here,
    # because imports can initialize CUDA or pull in CUDA-only libraries.
    return {
        "torch": _package_version("torch"),
        "torchvision": _package_version("torchvision"),
        "monai": _package_version("monai"),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "accelerate": _package_version("accelerate"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "torch_xla": _package_version("torch_xla"),
        "libtpu": _package_version("libtpu"),
    }


def _storage_section() -> dict[str, Any]:
    usage = shutil.disk_usage(Path.cwd())
    return {
        "free_disk_bytes": usage.free,
        "model_cache": _cache_status("MEDFM_MODEL_CACHE", Path.home() / ".cache" / "medfm" / "models"),
        "dataset_cache": _cache_status("MEDFM_DATASET_CACHE", Path.home() / ".cache" / "medfm" / "datasets"),
    }


def _nvidia_driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.splitlines()[0].strip() if out.stdout.strip() else None


def _cuda_section(torch: Any) -> dict[str, Any]:
    available = torch.cuda.is_available()
    devices: list[dict[str, Any]] = []
    if available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            free_bytes: int | None = None
            total_bytes = int(props.total_memory)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            except Exception:  # pragma: no cover - driver dependent
                pass
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_vram_bytes": total_bytes,
                    "free_vram_bytes": free_bytes,
                }
            )
    bf16 = bool(available and torch.cuda.is_bf16_supported())
    sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    flash_pkg = importlib.util.find_spec("flash_attn") is not None
    flash_sdp = bool(available and torch.backends.cuda.flash_sdp_enabled())
    nccl = bool(available and torch.distributed.is_available() and torch.distributed.is_nccl_available())
    return {
        "available": bool(available),
        "device_count": torch.cuda.device_count() if available else 0,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "driver_version": _nvidia_driver_version() if available else None,
        "devices": devices,
        "bf16_supported": bf16,
        "sdpa_available": bool(sdpa),
        "flash_attention": {"sdp_kernel": flash_sdp, "flash_attn_package": flash_pkg},
        "nccl_available": nccl,
    }


def _tpu_section() -> dict[str, Any]:
    """TPU diagnostics. Imported lazily; only runs for an explicit TPU request."""
    section: dict[str, Any] = {
        "torch_xla_version": _package_version("torch_xla"),
        "libtpu_version": _package_version("libtpu"),
        "pjrt_device": os.environ.get("PJRT_DEVICE"),
        "tpu_accelerator_type": os.environ.get("TPU_ACCELERATOR_TYPE"),
        "spmd_available": (
            importlib.util.find_spec("torch_xla") is not None
            and importlib.util.find_spec("torch_xla.distributed.spmd") is not None
        ),
        "incompatible_packages": [name for name in TPU_INCOMPATIBLE_PACKAGES if _package_version(name) is not None],
    }
    try:
        import torch  # noqa: PLC0415 - lazy by design
        import torch_xla.core.xla_model as xm  # noqa: PLC0415
        import torch_xla.runtime as xr  # noqa: PLC0415
    except ImportError as exc:
        section["available"] = False
        section["error"] = f"torch_xla not importable: {exc}"
        return section

    try:
        device_count = xr.global_runtime_device_count()
        section["available"] = True
        section["device_count"] = device_count
        section["device_type"] = xr.device_type()
        device = xm.xla_device()
        bf16_ok = True
        try:
            _ = (torch.zeros((1,), dtype=torch.bfloat16, device=device) + 1).item()
        except Exception:
            bf16_ok = False
        section["xla_bf16_supported"] = bf16_ok
    except Exception as exc:  # no TPU runtime attached
        section["available"] = False
        section["error"] = str(exc)
    return section


def _cpu_section() -> dict[str, Any]:
    return {"available": True, "physical_cores": os.cpu_count()}


def detect_backend() -> str:
    """Cheap backend detection that never initializes CUDA or XLA."""
    try:
        import torch  # noqa: PLC0415 - lazy by design

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU" or Path("/dev/accel0").exists():
        return "xla_tpu"
    return "cpu"


def collect(backend: str = "auto") -> dict[str, Any]:
    selected = detect_backend() if backend == "auto" else backend
    import torch  # noqa: PLC0415 - lazy: keeps `import medfm.tools.doctor` light

    accelerator: dict[str, Any]
    if selected == "cuda":
        accelerator = _cuda_section(torch)
    elif selected == "xla_tpu":
        accelerator = _tpu_section()
    else:
        accelerator = _cpu_section()

    return {
        "schema_version": SCHEMA_VERSION,
        "backend": selected,
        "python": _python_section(),
        "torch": {
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "cuda_initialized": bool(torch.cuda.is_initialized()),
        },
        "packages": _packages_section(),
        "storage": _storage_section(),
        "accelerator": accelerator,
    }


def _render_human(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"medfm doctor (schema v{report['schema_version']})")
    lines.append(f"  backend:            {report['backend']}")
    py = report["python"]
    lines.append(f"  python:             {py['version']} ({py['implementation']})")
    torch = report["torch"]
    lines.append(f"  torch:              {torch['version']} (cuda build: {torch['cuda_build_version']})")
    pkgs = report["packages"]
    for name in ("monai", "transformers", "peft", "bitsandbytes", "torch_xla", "libtpu"):
        lines.append(f"  {name:<20}{pkgs.get(name) or 'not installed'}")
    storage = report["storage"]
    lines.append(f"  free disk:          {storage['free_disk_bytes'] / 2**30:.1f} GiB")
    for key in ("model_cache", "dataset_cache"):
        cache = storage[key]
        lines.append(f"  {key:<20}{cache['path']} (writable: {cache['writable']})")
    acc = report["accelerator"]
    lines.append("  accelerator:")
    for key, value in acc.items():
        lines.append(f"    {key:<22}{value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="medfm runtime diagnostics")
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "cuda", "xla_tpu"],
        default="auto",
        help="accelerator backend to diagnose (default: auto)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = collect(backend=args.backend)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_human(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
