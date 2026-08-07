"""Unified batch schema: MedicalBatch, static-shape buckets, device transfer.

``modality`` is authoritative: tensor rank is validated *against* it and is
never used to infer it (a rank-5 tensor is a 3D volume, a multi-image set, or
a WSI tile set depending on the declared modality).

Bucket contract (ADR 0008): on static-shape paths, variable dimensions are
padded to a declared bucket and every padded dimension is covered by a mask —
``attention_mask`` for text tokens, ``image_mask`` for padded samples /
images / tiles, and ``task_targets["visual_token_mask"]`` for visual tokens.
A bucketed batch missing a required mask is a contract violation.

Device transfer is backend-neutral: ``to(device)`` and ``pin_memory()`` move
tensors only and preserve all non-tensor metadata. ``.cuda()`` and backend
imports (``torch_xla``, ``bitsandbytes``) are prohibited in this module by
policy and enforced by tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from medfm.core.enums import MODALITY_PIXEL_AXES, Modality, StrictStrEnum
from medfm.core.errors import BucketError, ShapeContractError
from medfm.core.sample import SpatialMetadata
from medfm.core.serialization import TensorMeta


class BucketKind(StrictStrEnum):
    """Which variable dimension a static-shape bucket covers."""

    IMAGE_2D = "IMAGE_2D"  # (H, W)
    VOLUME_3D = "VOLUME_3D"  # (D, H, W)
    MULTI_IMAGE = "MULTI_IMAGE"  # (I,) images per sample
    WSI_TILES = "WSI_TILES"  # (T,) tiles per slide
    VISUAL_TOKENS = "VISUAL_TOKENS"  # (N,) visual tokens per sample
    TEXT_TOKENS = "TEXT_TOKENS"  # (L,) text tokens per sample

    @property
    def rank(self) -> int:
        return _BUCKET_SHAPE_RANK[self]


_BUCKET_SHAPE_RANK: dict[BucketKind, int] = {
    BucketKind.IMAGE_2D: 2,
    BucketKind.VOLUME_3D: 3,
    BucketKind.MULTI_IMAGE: 1,
    BucketKind.WSI_TILES: 1,
    BucketKind.VISUAL_TOKENS: 1,
    BucketKind.TEXT_TOKENS: 1,
}


@dataclass(frozen=True)
class BucketId:
    """Identity of a fixed-shape bucket: which dimension, which padded shape.

    ``shape`` semantics by kind:
    - IMAGE_2D: ``(H, W)``
    - VOLUME_3D: ``(D, H, W)``
    - MULTI_IMAGE / WSI_TILES / VISUAL_TOKENS / TEXT_TOKENS: ``(count,)``
    """

    kind: BucketKind
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        shape = tuple(int(d) for d in self.shape)
        if len(shape) != self.kind.rank or any(d <= 0 for d in shape):
            raise BucketError(
                f"bucket {self.kind} requires a positive shape of rank {self.kind.rank}; got {self.shape}"
            )
        object.__setattr__(self, "shape", shape)

    def __str__(self) -> str:
        return f"{self.kind.value}:{'x'.join(str(d) for d in self.shape)}"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BucketId:
        return cls(kind=BucketKind.from_value(str(data["kind"])), shape=tuple(int(d) for d in data["shape"]))


def _is_mask(tensor: torch.Tensor) -> bool:
    return tensor.dtype == torch.bool or set(torch.unique(tensor.detach().cpu()).tolist()) <= {0, 1}


def _check_mask(tensor: torch.Tensor, name: str) -> None:
    if not _is_mask(tensor):
        raise ShapeContractError(f"{name} must be a boolean or 0/1 mask; got dtype {tensor.dtype}")


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class MedicalBatch:
    """Canonical collated batch.

    Tensor shape semantics (B = batch size):
    - ``pixel_values``: rank/axes per ``modality`` (see MODALITY_PIXEL_AXES);
      visual-token batches carry ``task_targets["visual_tokens"] [B, N, Dv]``
      instead.
    - ``image_mask``: real-vs-padded mask — ``[B]`` for single-image
      modalities, ``[B, I]`` / ``[B, T]`` / ``[B, S]`` for multi-image / WSI /
      multi-series batches.
    - ``tile_coordinates``: ``[B, T, 2|4]`` int level-0 slide pixels (WSI only).
    - ``input_ids``/``attention_mask``: ``[B, L]``.
    - ``labels``: ``[B]`` or ``[B, num_classes]``.
    - ``task_targets["segmentation"]``: ``[B, K, H, W]`` (2D) or
      ``[B, K, D, H, W]`` (3D), spatial dims matching ``pixel_values``.
    - ``task_targets["visual_tokens"] [B, N, Dv]`` with
      ``task_targets["visual_token_mask"] [B, N]``.
    """

    modality: Modality
    sample_ids: list[str]
    pixel_values: torch.Tensor | None = None
    image_mask: torch.Tensor | None = None
    tile_coordinates: torch.Tensor | None = None
    spatial_metadata: list[SpatialMetadata | None] = field(default_factory=list)
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    task_targets: dict[str, Any] = field(default_factory=dict)
    bucket: BucketId | None = None
    pinned: bool = False

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate(self) -> None:
        batch_size = self._infer_batch_size()
        if len(self.sample_ids) != batch_size:
            raise ShapeContractError(
                f"sample_ids has {len(self.sample_ids)} entries but the batch dimension is {batch_size}"
            )
        if self.spatial_metadata and len(self.spatial_metadata) != batch_size:
            raise ShapeContractError(
                f"spatial_metadata has {len(self.spatial_metadata)} entries but the batch dimension is {batch_size}"
            )
        self._validate_pixel_tensor()
        self._validate_image_mask()
        self._validate_tile_coordinates()
        self._validate_text_tensors()
        self._validate_labels(batch_size)
        self._validate_task_targets(batch_size)
        if self.bucket is not None:
            self._validate_bucket()

    def _infer_batch_size(self) -> int:
        if self.pixel_values is not None:
            return int(self.pixel_values.shape[0])
        if self.input_ids is not None:
            return int(self.input_ids.shape[0])
        visual = self.task_targets.get("visual_tokens")
        if isinstance(visual, torch.Tensor):
            return int(visual.shape[0])
        raise ShapeContractError(
            "batch carries neither pixel_values, input_ids, nor visual_tokens; cannot determine batch size"
        )

    def _validate_pixel_tensor(self) -> None:
        expected_rank = self.modality.expected_pixel_rank
        if expected_rank is None:
            if self.pixel_values is not None:
                raise ShapeContractError(
                    f"modality {self.modality} carries no pixel_values; got a tensor of shape "
                    f"{tuple(self.pixel_values.shape)}. TEXT is authoritative: use a non-text modality "
                    "for image batches."
                )
            return
        if self.pixel_values is None:
            if self.modality is Modality.PATHOLOGY_WSI and isinstance(
                self.task_targets.get("visual_tokens"), torch.Tensor
            ):
                return  # precomputed tile-embedding batches carry visual tokens instead
            raise ShapeContractError(
                f"modality {self.modality} requires pixel_values with rank {expected_rank} "
                f"(axes [{self._axes()}]); got None"
            )
        rank = self.pixel_values.ndim
        if rank != expected_rank:
            raise ShapeContractError(
                f"modality {self.modality} expects pixel_values rank {expected_rank} "
                f"(axes [{self._axes()}]); got rank {rank} with shape {tuple(self.pixel_values.shape)}. "
                "Fix the modality or the tensor — modality is never inferred from rank."
            )

    def _axes(self) -> str:
        axes = MODALITY_PIXEL_AXES[self.modality]
        return ", ".join(axes) if axes else ""

    def _validate_image_mask(self) -> None:
        if self.image_mask is None or self.pixel_values is None:
            return
        _check_mask(self.image_mask, "image_mask")
        b = int(self.pixel_values.shape[0])
        expected: tuple[int, ...]
        if self.modality in (Modality.MULTI_IMAGE_2D, Modality.PATHOLOGY_WSI, Modality.MULTI_SERIES_3D):
            n = int(self.pixel_values.shape[1])
            expected = (b, n)
        else:
            expected = (b,)
        if tuple(self.image_mask.shape) != expected:
            raise ShapeContractError(
                f"image_mask for {self.modality} must have shape {expected}; got {tuple(self.image_mask.shape)}"
            )

    def _validate_tile_coordinates(self) -> None:
        if self.tile_coordinates is None:
            return
        if not self.modality.is_pathology:
            raise ShapeContractError(f"tile_coordinates are only valid for pathology modalities; got {self.modality}")
        tc = self.tile_coordinates
        if tc.ndim != 3 or tc.shape[2] not in (2, 4):
            raise ShapeContractError(f"tile_coordinates must have shape [B, T, 2|4]; got {tuple(tc.shape)}")
        if tc.dtype not in (torch.int32, torch.int64):
            raise ShapeContractError(f"tile_coordinates must be int32/int64; got {tc.dtype}")
        if self.modality is Modality.PATHOLOGY_WSI and self.pixel_values is not None:
            if int(tc.shape[1]) != int(self.pixel_values.shape[1]):
                raise ShapeContractError(
                    f"tile_coordinates T={int(tc.shape[1])} does not match pixel_values "
                    f"T={int(self.pixel_values.shape[1])}"
                )

    def _validate_text_tensors(self) -> None:
        if self.modality.is_text_only and self.input_ids is None:
            raise ShapeContractError(f"modality {self.modality} requires input_ids [B, L]")
        if self.input_ids is not None:
            if self.input_ids.ndim != 2:
                raise ShapeContractError(f"input_ids must have shape [B, L]; got {tuple(self.input_ids.shape)}")
            if self.input_ids.dtype not in (torch.int32, torch.int64):
                raise ShapeContractError(f"input_ids must be int32/int64; got {self.input_ids.dtype}")
        if self.attention_mask is not None:
            if self.input_ids is None:
                raise ShapeContractError("attention_mask requires input_ids")
            if tuple(self.attention_mask.shape) != tuple(self.input_ids.shape):
                raise ShapeContractError(
                    f"attention_mask shape {tuple(self.attention_mask.shape)} must equal input_ids shape "
                    f"{tuple(self.input_ids.shape)}"
                )
            _check_mask(self.attention_mask, "attention_mask")

    def _validate_labels(self, batch_size: int) -> None:
        if self.labels is None:
            return
        if self.labels.ndim not in (1, 2) or int(self.labels.shape[0]) != batch_size:
            raise ShapeContractError(
                f"labels must have shape [B] or [B, num_classes] with B={batch_size}; got {tuple(self.labels.shape)}"
            )

    #: Modalities whose pixel grid can carry a dense segmentation target.
    _SEGMENTATION_MODALITIES = (
        Modality.XRAY_2D,
        Modality.CT_2D_SLICE,
        Modality.MRI_2D_SLICE,
        Modality.PATHOLOGY_TILE,
        Modality.CT_3D,
        Modality.MRI_3D,
        Modality.MULTI_SERIES_3D,
    )

    def _validate_task_targets(self, batch_size: int) -> None:
        segmentation = self.task_targets.get("segmentation")
        if segmentation is not None:
            if not isinstance(segmentation, torch.Tensor):
                raise ShapeContractError("task_targets['segmentation'] must be a tensor")
            self._validate_segmentation(segmentation, batch_size)
        visual = self.task_targets.get("visual_tokens")
        if visual is not None:
            if not isinstance(visual, torch.Tensor) or visual.ndim != 3:
                raise ShapeContractError(
                    f"task_targets['visual_tokens'] must be a [B, N, Dv] tensor; got {type(visual).__name__}"
                )
            if int(visual.shape[0]) != batch_size:
                raise ShapeContractError(f"visual_tokens batch dim {int(visual.shape[0])} != batch size {batch_size}")
            token_mask = self.task_targets.get("visual_token_mask")
            if token_mask is not None:
                if not isinstance(token_mask, torch.Tensor) or tuple(token_mask.shape) != tuple(visual.shape[:2]):
                    raise ShapeContractError(
                        f"visual_token_mask must have shape {tuple(visual.shape[:2])}; "
                        f"got {getattr(token_mask, 'shape', type(token_mask).__name__)}"
                    )
                _check_mask(token_mask, "visual_token_mask")

    def _validate_segmentation(self, segmentation: torch.Tensor, batch_size: int) -> None:
        if self.modality not in self._SEGMENTATION_MODALITIES:
            raise ShapeContractError(
                f"segmentation targets are not defined for modality {self.modality} "
                "(see the modality x task matrix in model_registry/v1_scope.yaml)"
            )
        if self.pixel_values is None:
            raise ShapeContractError("segmentation targets require pixel_values in the same batch")
        if self.modality in (Modality.CT_3D, Modality.MRI_3D, Modality.MULTI_SERIES_3D):
            expected_rank = 5  # [B, K, D, H, W]
            pixel_spatial = tuple(self.pixel_values.shape[-3:])
        else:
            expected_rank = 4  # [B, K, H, W]
            pixel_spatial = tuple(self.pixel_values.shape[-2:])
        if segmentation.ndim != expected_rank:
            axes = "B, K, D, H, W" if expected_rank == 5 else "B, K, H, W"
            raise ShapeContractError(
                f"segmentation for {self.modality} must have shape [{axes}]; got {tuple(segmentation.shape)}"
            )
        if int(segmentation.shape[0]) != batch_size or tuple(segmentation.shape[2:]) != pixel_spatial:
            raise ShapeContractError(
                f"segmentation spatial dims {tuple(segmentation.shape[2:])} must match pixel spatial dims "
                f"{pixel_spatial} with B={batch_size}"
            )

    def _validate_bucket(self) -> None:
        assert self.bucket is not None
        kind = self.bucket.kind
        shape = self.bucket.shape

        def fail(reason: str) -> None:
            raise BucketError(f"bucket {self.bucket}: {reason}")

        if kind is BucketKind.TEXT_TOKENS:
            if self.input_ids is None:
                fail("TEXT_TOKENS bucket requires input_ids")
            elif int(self.input_ids.shape[1]) != shape[0]:
                fail(f"padded text length {int(self.input_ids.shape[1])} != bucket shape {shape}")
            if self.attention_mask is None:
                fail("padded text tokens require attention_mask so padding is distinguishable")
        elif kind is BucketKind.VISUAL_TOKENS:
            visual = self.task_targets.get("visual_tokens")
            if not isinstance(visual, torch.Tensor):
                fail("VISUAL_TOKENS bucket requires task_targets['visual_tokens']")
            elif int(visual.shape[1]) != shape[0]:
                fail(f"padded visual-token count {int(visual.shape[1])} != bucket shape {shape}")
            if self.task_targets.get("visual_token_mask") is None:
                fail("padded visual tokens require task_targets['visual_token_mask']")
        elif kind is BucketKind.MULTI_IMAGE:
            if self.modality is not Modality.MULTI_IMAGE_2D or self.pixel_values is None:
                fail("MULTI_IMAGE bucket requires MULTI_IMAGE_2D pixel_values")
            elif int(self.pixel_values.shape[1]) != shape[0]:
                fail(f"padded image count {int(self.pixel_values.shape[1])} != bucket shape {shape}")
            if self.image_mask is None:
                fail("padded images require image_mask [B, I]")
        elif kind is BucketKind.WSI_TILES:
            if self.modality is not Modality.PATHOLOGY_WSI:
                fail("WSI_TILES bucket requires PATHOLOGY_WSI modality")
            if self.pixel_values is not None and int(self.pixel_values.shape[1]) != shape[0]:
                fail(f"padded tile count {int(self.pixel_values.shape[1])} != bucket shape {shape}")
            if self.image_mask is None:
                fail("padded tiles require image_mask [B, T]")
        elif kind is BucketKind.IMAGE_2D:
            if self.pixel_values is None or self.modality.expected_pixel_rank != 4:
                fail("IMAGE_2D bucket requires a rank-4 [B, C, H, W] modality")
            elif tuple(self.pixel_values.shape[-2:]) != shape:
                fail(f"padded image dims {tuple(self.pixel_values.shape[-2:])} != bucket shape {shape}")
        elif kind is BucketKind.VOLUME_3D:
            if self.pixel_values is None or self.modality not in (Modality.CT_3D, Modality.MRI_3D):
                fail("VOLUME_3D bucket requires CT_3D or MRI_3D pixel_values")
            elif tuple(self.pixel_values.shape[-3:]) != shape:
                fail(f"padded volume dims {tuple(self.pixel_values.shape[-3:])} != bucket shape {shape}")

    # ------------------------------------------------------------------ #
    # Device transfer (backend-neutral; no .cuda(), no backend imports)
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device | None:
        """Device of the batch's tensors, or None if the batch has no tensors."""
        for tensor in self._tensor_fields():
            if tensor is not None:
                return tensor.device
        return None

    def _tensor_fields(self) -> list[torch.Tensor | None]:
        visual = self.task_targets.get("visual_tokens")
        return [
            self.pixel_values,
            self.image_mask,
            self.tile_coordinates,
            self.input_ids,
            self.attention_mask,
            self.labels,
            visual if isinstance(visual, torch.Tensor) else None,
        ]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> MedicalBatch:
        """Return a copy with every tensor moved to ``device``.

        ``non_blocking`` is honored for tensor payloads when the source is
        pinned CPU memory; metadata follows the same device contract without
        changing its representation.
        """
        moved_targets = {
            key: (value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value)
            for key, value in self.task_targets.items()
        }
        return replace(
            self,
            pixel_values=(
                self.pixel_values.to(device, non_blocking=non_blocking) if self.pixel_values is not None else None
            ),
            image_mask=(self.image_mask.to(device, non_blocking=non_blocking) if self.image_mask is not None else None),
            tile_coordinates=(
                self.tile_coordinates.to(device, non_blocking=non_blocking)
                if self.tile_coordinates is not None
                else None
            ),
            spatial_metadata=[m.to(device) if m is not None else None for m in self.spatial_metadata],
            input_ids=(self.input_ids.to(device, non_blocking=non_blocking) if self.input_ids is not None else None),
            attention_mask=(
                self.attention_mask.to(device, non_blocking=non_blocking) if self.attention_mask is not None else None
            ),
            labels=(self.labels.to(device, non_blocking=non_blocking) if self.labels is not None else None),
            task_targets=moved_targets,
            pinned=self.pinned and torch.device(device).type == "cpu",
        )

    def pin_memory(self) -> MedicalBatch:
        """Return a copy with CPU tensors pinned for nonblocking host->device copies."""
        for tensor in self._tensor_fields():
            if tensor is not None and tensor.device.type != "cpu":
                raise ShapeContractError("pin_memory() requires CPU-resident tensors")
        pinned_targets = {
            key: (value.pin_memory() if isinstance(value, torch.Tensor) else value)
            for key, value in self.task_targets.items()
        }
        return replace(
            self,
            pixel_values=self.pixel_values.pin_memory() if self.pixel_values is not None else None,
            image_mask=self.image_mask.pin_memory() if self.image_mask is not None else None,
            tile_coordinates=(self.tile_coordinates.pin_memory() if self.tile_coordinates is not None else None),
            input_ids=self.input_ids.pin_memory() if self.input_ids is not None else None,
            attention_mask=self.attention_mask.pin_memory() if self.attention_mask is not None else None,
            labels=self.labels.pin_memory() if self.labels is not None else None,
            task_targets=pinned_targets,
            pinned=True,
        )

    # ------------------------------------------------------------------ #
    # Serialization: metadata only, never payloads, never devices
    # ------------------------------------------------------------------ #

    def tensor_metadata(self) -> dict[str, Any]:
        """Shape + accelerator-neutral dtype for every tensor field."""
        meta: dict[str, Any] = {}
        for name in (
            "pixel_values",
            "image_mask",
            "tile_coordinates",
            "input_ids",
            "attention_mask",
            "labels",
        ):
            tensor = getattr(self, name)
            if isinstance(tensor, torch.Tensor):
                meta[name] = TensorMeta.of(tensor).to_dict()
        for key, value in sorted(self.task_targets.items()):
            if isinstance(value, torch.Tensor):
                meta[f"task_targets.{key}"] = TensorMeta.of(value).to_dict()
        return meta

    def to_metadata_dict(self) -> dict[str, Any]:
        """Canonical non-tensor representation of the batch.

        Contains no tensor payloads and no device locations; tensors appear as
        :class:`TensorMeta` entries only.
        """
        return {
            "modality": self.modality.value,
            "sample_ids": list(self.sample_ids),
            "bucket": self.bucket.to_dict() if self.bucket is not None else None,
            "pinned": self.pinned,
            "spatial_metadata": [m.to_dict() if m is not None else None for m in self.spatial_metadata],
            "tensors": self.tensor_metadata(),
            "task_target_keys": sorted(self.task_targets),
        }
