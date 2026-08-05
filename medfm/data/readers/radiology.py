"""Radiology payload readers: NIfTI, MHA, NumPy volumes, PNG/JPEG.

All readers preserve geometry: affine, spacing, on-disk dtype, and
orientation are surfaced through :class:`SpatialMetadata` and never silently
dropped. Heavy backends (nibabel, SimpleITK, PIL) are imported lazily so the
module imports on a bare base install; missing backends fail at read time
with an actionable :class:`UnsupportedFormatError`.

Axis contract (see :mod:`medfm.data.readers.base`): volumetric arrays come
back in voxel order ``(i, j, k)`` with ``affine @ [i, j, k, 1]`` mapping to
patient-space millimetres.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medfm.core.sample import SpatialMetadata
from medfm.data.errors import ReaderError, UnsupportedFormatError
from medfm.data.readers.base import PayloadRead, Reader

#: Numpy dtype -> canonical torch dtype for reader outputs. uint16 has no
#: canonical torch name, so unsigned 16-bit payloads widen to int32.
_NUMPY_TO_TORCH: dict[np.dtype[Any], torch.dtype] = {
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.uint16): torch.int32,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
}


def _torch_from_numpy(array: np.ndarray[Any, Any]) -> torch.Tensor:
    dtype = _NUMPY_TO_TORCH.get(array.dtype)
    if dtype is None:
        raise UnsupportedFormatError(
            f"payload dtype {array.dtype} has no canonical torch mapping; cast the payload to one of "
            f"{sorted(str(d) for d in _NUMPY_TO_TORCH)} before ingestion"
        )
    buffer = np.ascontiguousarray(array)
    if not buffer.flags.writeable:
        buffer = buffer.copy()  # mmap/ro buffers: torch needs writable memory
    return torch.from_numpy(buffer).to(dtype)


def _require(module_name: str, extra: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise UnsupportedFormatError(
            f"reading this payload requires {module_name!r}; install medfm with the {extra!r} extra "
            f"(uv sync --extra {extra})"
        ) from exc


def _orientation_code(affine: np.ndarray[Any, Any]) -> str:
    nib = _require("nibabel", "medical")
    return "".join(nib.aff2axcodes(affine))


def _spatial_from_volume(
    array: np.ndarray[Any, Any],
    *,
    affine: np.ndarray[Any, Any] | None,
    spacing_mm: tuple[float, ...] | None,
    slice_positions_mm: torch.Tensor | None = None,
    frame_of_reference_hash: str | None = None,
    orientation: str | None = None,
) -> SpatialMetadata:
    shape = tuple(int(d) for d in array.shape)
    affine_tensor = torch.as_tensor(affine, dtype=torch.float64) if affine is not None else None
    if orientation is None and affine is not None:
        orientation = _orientation_code(affine)
    return SpatialMetadata(
        original_shape=shape,
        current_shape=shape,
        affine=affine_tensor,
        original_affine=affine_tensor,
        spacing_mm=spacing_mm,
        orientation=orientation,
        slice_positions_mm=slice_positions_mm,
        frame_of_reference_hash=frame_of_reference_hash,
    )


class NiftiReader(Reader):
    """NIfTI (.nii / .nii.gz) volumes via nibabel; affine/spacing/dtype/orientation preserved."""

    reader_id = "nifti"
    reader_version = "1.0.0"

    def supports(self, path: Path) -> bool:
        return path.name.endswith(".nii.gz") or path.suffix.lower() == ".nii"

    def read(self, path: Path) -> PayloadRead:
        nib = _require("nibabel", "medical")
        try:
            image = nib.load(str(path))
        except Exception as exc:
            raise ReaderError(f"cannot read NIfTI file {path}: {exc}") from exc
        # Preserve the ON-DISK dtype: get_fdata() upcasts to float64 and loses
        # the acquisition dtype, which preprocessing windows may depend on.
        array = np.asanyarray(image.dataobj, dtype=image.get_data_dtype())
        affine = np.asarray(image.affine, dtype=np.float64)
        zooms = tuple(float(z) for z in image.header.get_zooms())
        return PayloadRead(
            tensors={"image": _torch_from_numpy(array)},
            spatial=_spatial_from_volume(array, affine=affine, spacing_mm=zooms),
            source_metadata={"reader": self.reader_id, "reader_version": self.reader_version},
        )


class MHAReader(Reader):
    """MetaImage (.mha/.mhd) volumes via SimpleITK.

    SimpleITK arrays come back as ``(k, j, i)`` (z, y, x); they are
    transposed to the framework's ``(i, j, k)`` contract with the affine
    built from direction/spacing/origin.
    """

    reader_id = "mha"
    reader_version = "1.0.0"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in (".mha", ".mhd")

    def read(self, path: Path) -> PayloadRead:
        sitk = _require("SimpleITK", "medical")
        try:
            image = sitk.ReadImage(str(path))
        except Exception as exc:
            raise ReaderError(f"cannot read MHA file {path}: {exc}") from exc
        array_kji = sitk.GetArrayFromImage(image)
        array = np.ascontiguousarray(array_kji.transpose(2, 1, 0))  # (k,j,i) -> (i,j,k)
        spacing = tuple(float(s) for s in image.GetSpacing())  # (x, y, z)
        origin = tuple(float(o) for o in image.GetOrigin())
        direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = direction * np.asarray(spacing)
        affine[:3, 3] = origin
        return PayloadRead(
            tensors={"image": _torch_from_numpy(array)},
            spatial=_spatial_from_volume(array, affine=affine, spacing_mm=spacing),
            source_metadata={"reader": self.reader_id, "reader_version": self.reader_version},
        )


class NumpyVolumeReader(Reader):
    """Raw NumPy volumes (.npy) with an optional ``<name>.meta.json`` sidecar.

    The sidecar may carry ``affine`` (4x4 nested list) and/or ``spacing_mm``;
    without it the volume loads with geometry ``None`` (callers must not
    assume physical coordinates).
    """

    reader_id = "numpy_volume"
    reader_version = "1.0.0"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".npy"

    def read(self, path: Path) -> PayloadRead:
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise ReaderError(f"cannot read NumPy volume {path}: {exc}") from exc
        if not isinstance(array, np.ndarray) or array.ndim < 2:
            raise UnsupportedFormatError(f"{path} is not an array of rank >= 2; not a volume")
        meta: dict[str, Any] = {}
        sidecar = path.with_suffix(".meta.json")
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ReaderError(f"sidecar {sidecar.name} is not valid JSON: {exc.msg}") from exc
        affine = np.asarray(meta["affine"], dtype=np.float64) if "affine" in meta else None
        if affine is not None and affine.shape != (4, 4):
            raise ReaderError(f"sidecar affine must be 4x4; got shape {affine.shape}")
        spacing = tuple(float(s) for s in meta["spacing_mm"]) if "spacing_mm" in meta else None
        if spacing is not None and affine is None and len(spacing) != array.ndim:
            raise ReaderError(f"sidecar spacing_mm has {len(spacing)} entries; volume rank is {array.ndim}")
        if affine is None and spacing is None:
            # Geometry-free volume: SpatialMetadata still records shape.
            return PayloadRead(
                tensors={"image": _torch_from_numpy(array)},
                spatial=SpatialMetadata(
                    original_shape=tuple(int(d) for d in array.shape),
                    current_shape=tuple(int(d) for d in array.shape),
                ),
                source_metadata={"reader": self.reader_id, "reader_version": self.reader_version},
            )
        return PayloadRead(
            tensors={"image": _torch_from_numpy(array)},
            spatial=_spatial_from_volume(array, affine=affine, spacing_mm=spacing),
            source_metadata={"reader": self.reader_id, "reader_version": self.reader_version},
        )


class PngJpegReader(Reader):
    """Ordinary 2D images (PNG/JPEG) via Pillow.

    Grayscale stays 1-channel; RGB/RGBA reduce to RGB. Output tensor layout
    is ``(H, W, C)`` uint8 — preprocessing (Phase 04) owns channel policy.
    EXIF is stripped at decode time (it can carry identifiers).
    """

    reader_id = "png_jpeg"
    reader_version = "1.0.0"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in (".png", ".jpg", ".jpeg")

    def read(self, path: Path) -> PayloadRead:
        pil = _require("PIL.Image", "medical")
        try:
            with pil.open(path) as image:
                image.load()
                if image.mode in ("RGB", "RGBA"):
                    # .copy(): PIL hands back a read-only buffer; torch needs writable memory.
                    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                else:
                    array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        except Exception as exc:
            raise ReaderError(f"cannot decode image {path}: {exc}") from exc
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)  # (H, W) -> (H, W, 1)
        return PayloadRead(
            tensors={"image": tensor},
            source_metadata={"reader": self.reader_id, "reader_version": self.reader_version},
        )


def canonical_dtype_check(tensor: torch.Tensor) -> None:
    """Raise :class:`SerializationError` for non-canonical dtypes (used by tests)."""
    from medfm.core.serialization import canonical_dtype_name

    canonical_dtype_name(tensor.dtype)
