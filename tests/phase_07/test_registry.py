"""Registry and gated-integration checks for Phase 07."""

from __future__ import annotations

import pytest

from medfm.models.visual import (
    FlexiCT3DAdapter,
    FlexiCTVLMAdapter,
    M3DLaMedAdapter,
    TriadMAEAdapter,
    TriadSimMIMAdapter,
)
from medfm.registry import ModelRegistry, clear_plugins, get_plugin
from medfm.registry.catalog import load_v1_catalog


def setup_function() -> None:
    ModelRegistry.clear()
    clear_plugins()


def teardown_function() -> None:
    ModelRegistry.clear()
    clear_plugins()


def test_native_roster_has_separate_preprocess_and_plugins() -> None:
    specs = {spec.model_id: spec for spec in load_v1_catalog()}
    for model_id in ("ct-fm", "flexict-3d", "triad", "nv-segment-ctmr", "medsam2", "merlin", "m3d-lamed"):
        assert model_id in specs
        assert specs[model_id].preprocess is not None
        assert len(specs[model_id].preprocess.spatial_shape) == 3
        assert specs[model_id].capabilities.peft.known_target_modules
        assert get_plugin(model_id) is not None


def test_aliases_keep_flexict_and_triad_variants_separate() -> None:
    load_v1_catalog()
    assert ModelRegistry.get("ct_fm").model_id == "ct-fm"
    assert ModelRegistry.get("flexict_3d").model_id == "flexict-3d"
    assert ModelRegistry.get("flexict_3d_vlm").model_id == "flexict-3d"
    assert ModelRegistry.get("triad_mae").model_id == "triad"
    assert ModelRegistry.get("triad_simmim").model_id == "triad"


def test_variants_do_not_merge_capability_surfaces() -> None:
    mae = TriadMAEAdapter.build_tiny()
    simmim = TriadSimMIMAdapter.build_tiny()
    flexi = FlexiCT3DAdapter.build_tiny()
    vlm = FlexiCTVLMAdapter.build_tiny()
    assert mae.variant == "mae"
    assert simmim.variant == "simmim"
    assert flexi.model_id != vlm.model_id
    with pytest.raises(RuntimeError, match="gated"):
        M3DLaMedAdapter.build_tiny().generate()
