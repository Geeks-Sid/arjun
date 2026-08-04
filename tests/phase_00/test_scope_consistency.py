"""Consistency: modalities have backbones, tasks have implementation paths,
the modality x task matrix is a complete partition, scope matches licenses."""

from medfm.tools import governance as gov


def test_scope_is_consistent_with_licenses(scope, licenses):
    assert gov.check_scope_consistency(scope, set(licenses)) == []


def test_every_modality_has_a_backbone_candidate(scope):
    model_ids = {m["model_id"] for m in scope["models"]}
    for mod in scope["modalities"]:
        candidates = mod.get("backbone_candidates") or []
        assert candidates, f"{mod['name']} has no backbone candidate"
        assert set(candidates) <= model_ids, f"{mod['name']} lists unregistered candidates"
        assert mod["preferred_backbone"] in candidates
        assert mod["fallback_backbone"] in candidates


def test_every_supported_task_has_an_implementation_path(scope):
    matrix = scope["modality_task_matrix"]
    for task in scope["tasks"]:
        supported_somewhere = any(task["name"] in (matrix[m].get("supported") or []) for m in matrix)
        if supported_somewhere:
            assert task["implementation_path"], task["name"]
            assert task["primary_models"], f"{task['name']} supported but no primary model"


def test_matrix_is_complete_partition(scope):
    tasks = {t["name"] for t in scope["tasks"]}
    for mod in scope["modalities"]:
        row = scope["modality_task_matrix"][mod["name"]]
        all_entries = row["supported"] + row["deferred"] + row["unsupported"]
        assert len(all_entries) == len(set(all_entries)), f"{mod['name']} has duplicate dispositions"
        assert set(all_entries) == tasks, f"{mod['name']} does not cover every task"


def test_every_scope_model_has_license_record(scope, licenses):
    for model in scope["models"]:
        assert model["model_id"] in licenses, model["model_id"]


def test_no_orphan_license_records(scope, licenses):
    model_ids = {m["model_id"] for m in scope["models"]}
    for lic_id in licenses:
        assert lic_id in model_ids, lic_id


def test_vertical_slices_cover_required_families(scope):
    families = {vs["id"] for vs in scope["vertical_slices"]}
    assert {"VS-2D", "VS-3D", "VS-WSI", "VS-SEG", "VS-RETRIEVAL", "VS-VLM"} <= families


def test_model_task_claims_are_bidirectional(scope):
    models = {m["model_id"]: m for m in scope["models"]}
    for task in scope["tasks"]:
        for pm in task.get("primary_models") or []:
            assert task["name"] in models[pm]["tasks"], f"{task['name']} claims {pm} but {pm} does not list it"
