from __future__ import annotations

import pytest
import torch

from medfm.inference import (
    AdapterManager,
    AuditLogger,
    BundleBuilder,
    ClassificationPipeline,
    InferenceLimits,
    InferenceService,
)


def test_service_validates_before_adapter_loading_and_redacts_audit(bundle_builder, tmp_path) -> None:
    bundle_builder.add_adapter("adapter-a", {"weight": torch.ones(1)})
    bundle = bundle_builder.build()
    loads: list[str] = []

    def adapter_loader(base, loaded_bundle, name):
        del base, loaded_bundle
        loads.append(name)
        return lambda value: torch.zeros(value.shape[0], 2)

    manager = AdapterManager(adapter_loader=adapter_loader, max_loaded=1, allowed_bundle_root=tmp_path)
    manager.register("adapter-a", bundle.root)
    pipeline = ClassificationPipeline(
        lambda value: torch.zeros(value.shape[0], 2),
        limits=InferenceLimits(max_batch_size=1, max_image_pixels=16),
        model_id="m",
        model_revision="r",
        preprocess_hash="p",
    )
    audit_path = tmp_path / "operational.jsonl"
    service = InferenceService(
        {"classification": pipeline},
        adapter_manager=manager,
        audit_logger=AuditLogger(audit_path),
        limits=InferenceLimits(max_batch_size=1, max_image_pixels=16),
        queue_capacity=0,
    )
    bad = service.handle(
        {
            "task": "classification",
            "modality": "XRAY_2D",
            "adapter": "adapter-a",
            "payload": {"pixel_values": torch.zeros(1, 1, 5, 5), "report": "sensitive report"},
        }
    )
    assert bad.ok is False
    assert loads == []
    good = service.handle(
        {
            "task": "classification",
            "modality": "XRAY_2D",
            "adapter": "adapter-a",
            "payload": {"pixel_values": torch.zeros(1, 1, 4, 4), "report": "sensitive report"},
            "request_id": "safe-id",
        }
    )
    assert good.ok is True
    assert loads == ["adapter-a"]
    line = audit_path.read_text(encoding="utf-8")
    assert "sensitive report" not in line
    assert "input_hash" in line
    service.close()


def test_adapter_manager_switches_and_evicts_without_stale_state(bundle_builder, tmp_path) -> None:
    first = bundle_builder.build()
    # The same metadata shape is sufficient for a second registration; each
    # bundle is still independently checksum validated.
    second_builder = BundleBuilder(
        tmp_path / "bundle-b",
        bundle_id="b",
        model_id="m",
        model_revision="r",
        task="classification",
        base_models=first.manifest.base_models,
        model_card="card",
        license_summary="license",
        preprocessing={},
        postprocessing={},
        task_schema={"type": "object"},
        inference_config={},
        runtime={"backends": {"cpu": "tested"}},
    )
    second_builder.add_adapter("adapter-b", {"weight": torch.ones(1)})
    second = second_builder.build()
    manager = AdapterManager(
        base_model=object(),
        adapter_loader=lambda base, bundle, name: {"name": name, "base": base},
        max_loaded=1,
        allowed_bundle_root=tmp_path,
    )
    manager.register("adapter-a", first.root)
    manager.register("adapter-b", second.root)
    a = manager.load("adapter-a")
    b = manager.load("adapter-b")
    assert a["name"] == "adapter-a"
    assert b["name"] == "adapter-b"
    assert manager.loaded_adapters == ("adapter-b",)
    assert manager.active_adapter == "adapter-b"


def test_clinical_audit_store_requires_role_and_supports_deletion(tmp_path) -> None:
    from medfm.inference import ClinicalAuditStore

    store = ClinicalAuditStore(tmp_path / "clinical")
    with pytest.raises(PermissionError):
        store.put("e1", {"timestamp": "2020-01-01T00:00:00+00:00", "report": "restricted"}, role="operator")
    store.put("e1", {"timestamp": "2020-01-01T00:00:00+00:00", "report": "restricted"}, role="clinical_auditor")
    assert store.get("e1", role="clinical_auditor")["report"] == "restricted"
    assert store.delete("e1", role="clinical_auditor") is True
