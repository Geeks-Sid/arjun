"""Weight acquisition and integrity for the model registry.

Policy:
- pinned revisions only (enforced by ModelSpec);
- safetensors preferred; pickle-based formats rejected unless explicitly
  reviewed (``allow_unsafe_formats``);
- file hashes and expected file sets verified after download;
- partial downloads detected and rejected;
- tokens never appear in commands, logs, or manifests;
- ``trust_remote_code`` defaults to false: Python files are excluded from
  downloads unless the model carries a reviewed allowlist exception;
- gated repositories require a pre-recorded acceptance (see acceptance.py);
  downloading never implies acceptance.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from medfm.registry import acceptance
from medfm.registry.schema import ModelSpec

logger = logging.getLogger(__name__)

SAFE_EXTENSIONS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "*.txt",
    "*.model",  # SentencePiece models
]

UNSAFE_EXTENSIONS = [
    "*.bin",
    "*.pt",
    "*.pth",
    "*.msgpack",
    "*.h5",
    "*.pkl",
    "*.pickle",
]

#: huggingface_hub marks in-flight blobs with this suffix.
PARTIAL_SUFFIX = ".incomplete"


class GatedAccessError(RuntimeError):
    """Raised when weights require provider terms that were not accepted."""


class WeightIntegrityError(RuntimeError):
    """Raised on hash mismatch, missing/extra files, or partial downloads."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_partial_downloads(local_dir: Path | str) -> list[Path]:
    """In-flight or interrupted downloads leave ``*.incomplete`` blobs."""
    path = Path(local_dir)
    if not path.exists():
        return []
    return sorted(p for p in path.rglob(f"*{PARTIAL_SUFFIX}") if p.is_file())


def verify_file_hashes(local_dir: Path | str, expected: dict[str, str]) -> None:
    """Verify the exact expected file set and sha256 of each file.

    ``expected`` maps repo-relative paths to lowercase hex sha256 digests.
    Missing files, unexpected extra weight files, and hash mismatches raise
    WeightIntegrityError.
    """
    path = Path(local_dir)
    errors: list[str] = []
    for rel, digest in sorted(expected.items()):
        f = path / rel
        if not f.is_file():
            errors.append(f"missing expected file: {rel}")
            continue
        actual = sha256_file(f)
        if actual.lower() != digest.lower():
            errors.append(f"hash mismatch: {rel} (expected {digest}, got {actual})")

    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    for f in path.rglob("*"):
        if f.is_file() and f.suffix in weight_suffixes:
            rel = f.relative_to(path).as_posix()
            if rel not in expected:
                errors.append(f"unexpected weight file not in expected set: {rel}")

    if errors:
        raise WeightIntegrityError("; ".join(errors))


def verify_weight_integrity(local_dir: Path | str) -> bool:
    """Cheap structural check: directory exists, has files, no partial blobs."""
    path = Path(local_dir)
    if not path.exists() or not path.is_dir():
        return False
    if find_partial_downloads(path):
        logger.warning(f"partial download detected in {path}")
        return False
    files = [f for f in path.rglob("*") if f.is_file()]
    if not files:
        return False

    safetensors_exist = any(f.suffix == ".safetensors" for f in files)
    bin_exist = any(f.suffix == ".bin" for f in files)
    if safetensors_exist and bin_exist:
        logger.warning(f"Found both safetensors and unsafe .bin formats in {path}")
    return True


def inspect_weights(local_dir: Path | str) -> dict[str, object]:
    """Structured local-weight report: file set, sizes, formats, integrity."""
    path = Path(local_dir)
    files = []
    if path.exists():
        for f in sorted(path.rglob("*")):
            if f.is_file():
                files.append(
                    {
                        "path": f.relative_to(path).as_posix(),
                        "size_bytes": f.stat().st_size,
                        "format": f.suffix.lstrip("."),
                    }
                )
    partials = [p.relative_to(path).as_posix() for p in find_partial_downloads(path)]
    return {
        "local_dir": str(path),
        "exists": path.exists(),
        "files": files,
        "partial_downloads": partials,
        "integrity_ok": bool(files) and not partials,
        "uses_safetensors": any(f["format"] == "safetensors" for f in files),
        "uses_pickle_formats": any(f["format"] in {"bin", "pt", "pth", "pkl", "pickle"} for f in files),
    }


def resolve_local_path(spec: ModelSpec, cache_dir: Path | str) -> Path:
    """Resolve an already-downloaded weight directory without network access.

    Raises FileNotFoundError if the pinned revision is not present locally —
    callers must use ``download_weights`` (explicit, opt-in) to fetch.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("huggingface_hub is required for weight resolution.") from e

    local = snapshot_download(
        repo_id=spec.repository,
        revision=spec.revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    path = Path(local)
    if not verify_weight_integrity(path):
        raise WeightIntegrityError(f"local weights for {spec.model_id} failed integrity check")
    return path


def download_weights(
    spec: ModelSpec,
    cache_dir: Path | str,
    token: str | None = None,
    allow_unsafe_formats: bool = False,
    acceptance_store: Path | None = None,
) -> Path:
    """Download model weights safely at the pinned revision.

    Network access is explicit and opt-in: only this function downloads.
    Gated repositories require acceptance recorded beforehand via
    ``medfm.registry.acceptance.record_acceptance``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("huggingface_hub is required for weight downloads.") from e

    if spec.license.acceptance_required and not acceptance.has_accepted(
        spec.model_id, spec.repository, store_path=acceptance_store
    ):
        raise GatedAccessError(
            f"{spec.model_id} requires explicit acceptance of provider terms at "
            f"{spec.license.terms_url}. Record it with "
            f"`medfm models accept-terms {spec.model_id} --by <name>` first."
        )

    allow_patterns = SAFE_EXTENSIONS.copy()
    if allow_unsafe_formats:
        allow_patterns.extend(UNSAFE_EXTENSIONS)

    try:
        # Atomic download + resume are handled by huggingface_hub; the token is
        # passed only to the library and never to commands, logs, or manifests.
        local_dir = snapshot_download(
            repo_id=spec.repository,
            revision=spec.revision,
            cache_dir=str(cache_dir),
            token=token,
            allow_patterns=allow_patterns,
            local_files_only=False,
            ignore_patterns=["*.py"] if not spec.trust_remote_code_allowed else None,
        )
    except Exception as e:
        # Exception type only: provider errors can embed request URLs/tokens.
        raise RuntimeError(f"Failed to download weights for {spec.model_id}: {type(e).__name__}") from e

    path = Path(local_dir)
    partials = find_partial_downloads(path)
    if partials:
        raise WeightIntegrityError(f"partial download for {spec.model_id}: {[str(p) for p in partials]}")
    return path
