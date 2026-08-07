"""Backend neutrality and static-code checks for Phase 06 adapters."""

from pathlib import Path

from medfm.models.visual.raddino import RADDINOAdapter

VISUAL_DIR = Path(__file__).resolve().parents[2] / "medfm" / "models" / "visual"


def _source_files():
    return list(VISUAL_DIR.glob("*.py"))


def _read_all():
    return "\n".join(f.read_text() for f in _source_files())


def test_no_dot_cuda():
    text = _read_all()
    # Skip backtick-quoted docstring mentions and comments
    code = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#") and not line.strip().startswith('"')
    )
    assert ".cuda(" not in code, "found .cuda() in visual package"


def test_no_torch_xla_import():
    text = _read_all()
    assert "import torch_xla" not in text and "from torch_xla" not in text, "torch_xla in visual package"


def test_no_bitsandbytes_import():
    text = _read_all()
    assert "import bitsandbytes" not in text and "from bitsandbytes" not in text, "bitsandbytes in visual package"


def test_tpu_smoke_config_available():
    rd = RADDINOAdapter.build_tiny()
    cfg = rd.tpu_smoke_config()
    for key in ("model_id", "batch_size", "image_size", "dtype", "static_batch", "attention"):
        assert key in cfg
    assert cfg["static_batch"] is True
