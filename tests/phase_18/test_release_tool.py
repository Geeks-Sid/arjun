from __future__ import annotations

import hashlib

from medfm.tools import release


def test_registry_backend_statuses_resolvable() -> None:
    """Every catalog model resolves all five backends to a legal status."""
    assert release.registry_backend_statuses() == []


def test_registry_models_expose_all_five_backends() -> None:
    from medfm.registry import ModelRegistry, catalog

    catalog.ensure_v1_catalog()
    specs = ModelRegistry.list_models(include_blocked=True, include_deprecated=True)
    assert specs
    for spec in specs:
        assert set(spec.backend_support) == set(release.BACKEND_KEYS)


def test_license_registry_consistency_clean() -> None:
    assert release.license_registry_consistency() == []


def test_no_eager_backend_imports_clean() -> None:
    assert release.no_eager_backend_imports() == []


def test_eager_hostile_import_is_flagged(monkeypatch, tmp_path) -> None:
    package = tmp_path / "medfm" / "core"
    package.mkdir(parents=True)
    (package / "evil.py").write_text("import torch_xla\n\ndef f() -> None: ...\n", encoding="utf-8")
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release, "BACKEND_NEUTRAL_PACKAGES", ("medfm/core",))
    errors = release.no_eager_backend_imports()
    assert any("torch_xla" in message and "evil.py" in message for message in errors)


def test_guarded_hostile_import_is_allowed(monkeypatch, tmp_path) -> None:
    package = tmp_path / "medfm" / "core"
    package.mkdir(parents=True)
    (package / "ok.py").write_text(
        "def f() -> None:\n    import torch_xla  # noqa: F401 (lazy)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release, "BACKEND_NEUTRAL_PACKAGES", ("medfm/core",))
    assert release.no_eager_backend_imports() == []


def test_tpu_nf4_config_flagged(monkeypatch, tmp_path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir(parents=True)
    (configs / "bad.yaml").write_text(
        "accelerator:\n  backend: xla_tpu\npeft:\n  quantization: {method: nf4}\n",
        encoding="utf-8",
    )
    (configs / "good.yaml").write_text(
        "accelerator:\n  backend: cpu\npeft:\n  quantization: {method: nf4}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    errors = release.no_tpu_nf4()
    assert any("bad.yaml" in message for message in errors)
    assert not any("good.yaml" in message for message in errors)


def test_clinical_claim_without_disclaimer_flagged(monkeypatch, tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    (docs / "claim.md").write_text("This system can diagnose acute findings.\n", encoding="utf-8")
    (docs / "labeled.md").write_text("This system can diagnose acute findings. Research use only.\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Framework for medical imaging research.\n", encoding="utf-8")
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    errors = release.clinical_claims()
    assert any("claim.md" in message for message in errors)
    assert not any("labeled.md" in message for message in errors)


def test_checksums_deterministic(tmp_path) -> None:
    target = tmp_path / "artifacts"
    target.mkdir()
    (target / "a.txt").write_text("alpha\n", encoding="utf-8")
    (target / "b.bin").write_bytes(b"\x00\x01\x02")
    first = release.checksums(target)
    second = release.checksums(target)
    assert first == second
    assert first["a.txt"] == hashlib.sha256(b"alpha\n").hexdigest()


def test_write_checksums_format(tmp_path) -> None:
    target = tmp_path / "artifacts"
    target.mkdir()
    (target / "a.txt").write_text("alpha\n", encoding="utf-8")
    out = release.write_checksums(target, out_path=tmp_path / "checksums.txt")
    text = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 1
    digest, relpath = text[0].split("  ", 1)
    assert relpath == "a.txt"
    assert int(digest, 16) > 0


def test_generate_support_matrix_covers_models(tmp_path) -> None:
    path = release.generate_support_matrix(out_path=tmp_path / "support_matrix.md")
    text = path.read_text(encoding="utf-8")
    from medfm.registry import ModelRegistry, catalog

    catalog.ensure_v1_catalog()
    specs = ModelRegistry.list_models(include_blocked=True, include_deprecated=True)
    assert any(f"| {spec.model_id} |" in text for spec in specs)
    for key in release.BACKEND_KEYS:
        assert key in text
