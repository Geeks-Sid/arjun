"""Phase 06 registry and CLI integration tests."""

import io
import json
import sys

import pytest

from medfm.cli.models import main as cli_main
from medfm.core.enums import LoadingMode
from medfm.registry import ModelRegistry, clear_plugins, get_plugin
from medfm.registry.catalog import load_v1_catalog
from medfm.registry.schema import BACKEND_KEYS, LicenseClass, ModelStatus


@pytest.fixture(autouse=True)
def _clean():
    ModelRegistry.clear()
    clear_plugins()
    yield
    clear_plugins()
    ModelRegistry.clear()


def _run(*args):
    try:
        return cli_main(list(args))
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 99


def _captured(*args):
    old = sys.stdout
    sys.stdout = io.StringIO()
    rc = _run(*args)
    out = sys.stdout.getvalue()
    sys.stdout = old
    return rc, out


# ------------------------------------------------------------------ #
# Registry
# ------------------------------------------------------------------ #


def test_raddino_ready_deployment():
    load_v1_catalog()
    rd = ModelRegistry.get("rad-dino")
    assert rd.status == ModelStatus.READY
    assert rd.license.class_type == LicenseClass.DEPLOYMENT
    assert len(rd.revision) == 40


def test_medsiglip_blocked():
    load_v1_catalog()
    ms = ModelRegistry.get("medsiglip")
    assert ms.status == ModelStatus.BLOCKED
    assert "license" in ms.blocked_reason.lower()  # type: ignore[union-attr]


def test_hoptimus_blocked():
    load_v1_catalog()
    ho = ModelRegistry.get("h-optimus-0")
    assert ho.status == ModelStatus.BLOCKED


def test_medgemma_blocked():
    load_v1_catalog()
    mg = ModelRegistry.get("medgemma-1.5-4b")
    assert mg.status == ModelStatus.BLOCKED


def test_medsiglip_aliases():
    load_v1_catalog()
    ms = ModelRegistry.get("medsiglip")
    assert "medsiglip_448" in ms.aliases


def test_medsiglip_alias_resolves():
    load_v1_catalog()
    ms = ModelRegistry.get("medsiglip_448")
    assert ms.model_id == "medsiglip"


def test_conch_no_plugin():
    load_v1_catalog()
    assert get_plugin("conch") is None


def test_all_2d_adapters_have_plugins():
    load_v1_catalog()
    for mid in ("medsiglip", "rad-dino", "h-optimus-0", "medgemma-1.5-4b"):
        assert get_plugin(mid) is not None, f"{mid} has no plugin"


def test_raddino_peft_targets():
    load_v1_catalog()
    rd = ModelRegistry.get("rad-dino")
    assert len(rd.capabilities.peft.known_target_modules) > 0


def test_medsiglip_preprocess_set():
    load_v1_catalog()
    ms = ModelRegistry.get("medsiglip")
    assert ms.preprocess is not None
    assert ms.preprocess.spatial_shape == (448, 448)


def test_backend_support_default_untested():
    load_v1_catalog()
    rd = ModelRegistry.get("rad-dino")
    for key in BACKEND_KEYS:
        assert rd.backend_support[key].status.value == "UNTESTED"


def test_tpu_nf4_rejected():
    load_v1_catalog()
    with pytest.raises(ValueError, match="bitsandbytes"):
        ModelRegistry.validate_backend(ModelRegistry.get("medgemma-1.5-4b"), "tpu_single_host", LoadingMode.QLORA_NF4)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #


def test_cli_smoke_raddino():
    load_v1_catalog()
    rc = _run("smoke", "rad-dino", "--backend", "cpu")
    assert rc == 0


def test_cli_smoke_medsiglip_blocked():
    load_v1_catalog()
    rc = _run("smoke", "medsiglip_448")
    assert rc != 0


def test_cli_ready_only():
    load_v1_catalog()
    rc, out = _captured("list", "--ready-only")
    assert rc == 0
    assert "rad-dino" in out


def test_cli_show_json():
    load_v1_catalog()
    rc, out = _captured("show", "rad-dino", "--format", "json")
    assert rc == 0
    data = json.loads(out)
    assert data["model_id"] == "rad-dino"
    assert len(data["revision"]) == 40


def test_cli_inspect_modules():
    load_v1_catalog()
    rc, out = _captured("inspect-modules", "rad-dino")
    assert rc == 0
    assert "target" in out.lower() or "known" in out.lower()


def test_cli_modality_filter():
    load_v1_catalog()
    rc, out = _captured("list", "--modality", "XRAY_2D")
    assert rc == 0
    assert "medsiglip" in out
    assert "rad-dino" in out
