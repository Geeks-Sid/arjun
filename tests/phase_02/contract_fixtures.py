"""Contract fixtures for Phase 02: synthetic samples/batches and dummy
protocol-conformant components (encoder, language adapter, task module).

These dummies are the static conformance fixtures handed to model authors in
later phases: any real adapter must satisfy the same protocols and output
semantics.
"""

from __future__ import annotations

from typing import Any

import torch

from medfm.core import (
    CoordinateSystem,
    EncoderCapabilities,
    EncoderOutput,
    GeneratedText,
    GenerationConfig,
    ImageReference,
    LabelTarget,
    LanguageModelCapabilities,
    LanguageOutput,
    LossOutput,
    MedicalBatch,
    MedicalSample,
    Modality,
    PathologyMetadata,
    PreprocessSpec,
    ProjectedVisualTokens,
    ProvenanceMetadata,
    SpatialMetadata,
    SplitName,
    TaskType,
    TokenizedText,
    UnsupportedCapabilityError,
)

HASH64 = "ab12" * 16  # valid 64-char lowercase hex digest
HASH32 = "cd34" * 8

VOCAB_SIZE = 32
LM_HIDDEN = 8


def make_provenance() -> ProvenanceMetadata:
    return ProvenanceMetadata(
        dataset_name="synthetic",
        dataset_version="0.1",
        split=SplitName.TRAIN,
        license="CC0-1.0",
        deidentification_method="hash-shift-remove",
    )


def make_spatial(shape: tuple[int, ...] = (16, 32, 32)) -> SpatialMetadata:
    rank = len(shape)
    affine = torch.diag(torch.tensor([2.0] * rank + [1.0], dtype=torch.float64))
    return SpatialMetadata(
        original_shape=shape,
        current_shape=shape,
        affine=affine,
        original_affine=affine.clone(),
        spacing_mm=tuple([2.0] * rank),
        orientation="RAS",
        anatomical_axes=tuple("RAS"[:rank]),
        slice_positions_mm=torch.arange(float(shape[0])) if rank == 3 else None,
        frame_of_reference_hash=HASH32,
    )


def make_pathology(num_tiles: int = 4) -> PathologyMetadata:
    return PathologyMetadata(
        microns_per_pixel=0.25,
        magnification=40.0,
        slide_dimensions=(100000, 80000),
        level_dimensions=((100000, 80000), (25000, 20000), (6250, 5000)),
        stain="H&E",
        scanner_vendor="synthetic-scan",
        tile_coordinates=torch.arange(num_tiles * 2, dtype=torch.int64).reshape(num_tiles, 2) * 256,
    )


def make_sample(modality: Modality) -> MedicalSample:
    """Construct a structurally valid synthetic sample for any modality."""
    kwargs: dict[str, Any] = {
        "sample_id": f"sample-{modality.value.lower()}",
        "patient_id_hash": HASH64,
        "study_id_hash": HASH64[::-1],
        "modality": modality,
        "provenance": make_provenance(),
    }
    if modality.is_text_only:
        kwargs["report"] = "No acute cardiopulmonary abnormality."
        kwargs["question"] = "Is there pneumonia?"
        kwargs["answer"] = "No."
        return MedicalSample(**kwargs)

    n_images = 3 if modality is Modality.MULTI_IMAGE_2D else 1
    if modality is Modality.MULTI_SERIES_3D:
        n_images = 2
    kwargs["image_references"] = tuple(
        ImageReference(uri=f"s3://bucket/{modality.value.lower()}-{i}.dcm", sha256=HASH64) for i in range(n_images)
    )
    if modality.is_pathology:
        kwargs["pathology"] = make_pathology()
    elif modality.is_volumetric:
        kwargs["spatial"] = make_spatial()
    kwargs["labels"] = LabelTarget(task=TaskType.BINARY_CLASSIFICATION, values=(1.0,))
    return MedicalSample(**kwargs)


def make_batch(modality: Modality, batch_size: int = 2) -> MedicalBatch:
    """Construct a structurally valid synthetic batch for any modality."""
    sample_ids = [f"s{i}" for i in range(batch_size)]
    if modality.is_text_only:
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, 8))
        return MedicalBatch(
            modality=modality,
            sample_ids=sample_ids,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )

    b = batch_size
    kwargs: dict[str, Any] = {"modality": modality, "sample_ids": sample_ids}
    if modality in (Modality.XRAY_2D, Modality.CT_2D_SLICE, Modality.MRI_2D_SLICE):
        kwargs["pixel_values"] = torch.randn(b, 1, 16, 16)
        kwargs["image_mask"] = torch.ones(b, dtype=torch.bool)
    elif modality in (Modality.CT_3D, Modality.MRI_3D):
        kwargs["pixel_values"] = torch.randn(b, 1, 8, 16, 16)
        kwargs["image_mask"] = torch.ones(b, dtype=torch.bool)
        kwargs["spatial_metadata"] = [make_spatial((8, 16, 16)) for _ in range(b)]
    elif modality is Modality.PATHOLOGY_TILE:
        kwargs["pixel_values"] = torch.randn(b, 3, 16, 16)
        kwargs["image_mask"] = torch.ones(b, dtype=torch.bool)
    elif modality is Modality.PATHOLOGY_WSI:
        kwargs["pixel_values"] = torch.randn(b, 4, 3, 16, 16)
        kwargs["image_mask"] = torch.ones(b, 4, dtype=torch.bool)
        kwargs["tile_coordinates"] = torch.zeros(b, 4, 2, dtype=torch.int64)
    elif modality is Modality.MULTI_IMAGE_2D:
        kwargs["pixel_values"] = torch.randn(b, 3, 3, 16, 16)
        kwargs["image_mask"] = torch.ones(b, 3, dtype=torch.bool)
    elif modality is Modality.MULTI_SERIES_3D:
        kwargs["pixel_values"] = torch.randn(b, 2, 1, 8, 16, 16)
        kwargs["image_mask"] = torch.ones(b, 2, dtype=torch.bool)
        kwargs["spatial_metadata"] = [make_spatial((8, 16, 16)) for _ in range(b)]
    else:  # pragma: no cover - exhaustive over the enum
        raise AssertionError(f"unhandled modality {modality}")
    return MedicalBatch(**kwargs)


class DummyVisualEncoder:
    """Protocol-conformant encoder over 2D grayscale images."""

    @property
    def capabilities(self) -> EncoderCapabilities:
        return EncoderCapabilities(
            model_id="dummy-2d-encoder",
            modalities=(Modality.XRAY_2D, Modality.CT_2D_SLICE, Modality.MRI_2D_SLICE),
            supports_pooled=True,
            supports_spatial_tokens=True,
            supports_feature_maps=False,
            supports_token_coordinates=True,
            token_coordinate_systems=(CoordinateSystem.NORMALIZED_IMAGE,),
        )

    def preprocess_spec(self) -> PreprocessSpec:
        return PreprocessSpec(image_size=(16, 16), channels=1)

    def encode(self, batch: MedicalBatch, output_hidden_states: bool = False) -> EncoderOutput:
        self.capabilities.require_modality(batch.modality)
        pixels = batch.pixel_values
        assert pixels is not None
        b, _, h, w = pixels.shape
        n = (h // 4) * (w // 4)
        tokens = torch.randn(b, n, 6)
        grid = torch.stack(
            torch.meshgrid(torch.linspace(0, 1, h // 4), torch.linspace(0, 1, w // 4), indexing="ij"),
            dim=-1,
        ).reshape(1, n, 2)
        return EncoderOutput(
            pooled_embedding=tokens.mean(dim=1),
            spatial_tokens=tokens,
            token_mask=torch.ones(b, n, dtype=torch.bool),
            token_coordinates=grid.expand(b, n, 2),
            token_coordinate_system=CoordinateSystem.NORMALIZED_IMAGE,
        )


class PoolingOnlyEncoder:
    """Encoder that cannot produce spatial tokens; must refuse, not pool."""

    @property
    def capabilities(self) -> EncoderCapabilities:
        return EncoderCapabilities(
            model_id="pooling-only-encoder",
            modalities=(Modality.XRAY_2D,),
            supports_pooled=True,
            supports_spatial_tokens=False,
        )

    def preprocess_spec(self) -> PreprocessSpec:
        return PreprocessSpec(image_size=(16, 16), channels=1)

    def encode(self, batch: MedicalBatch, output_hidden_states: bool = False) -> EncoderOutput:
        self.capabilities.require_modality(batch.modality)
        assert batch.pixel_values is not None
        return EncoderOutput(pooled_embedding=torch.zeros(batch.pixel_values.shape[0], 4))


class DummyLanguageModelAdapter:
    """Protocol-conformant LM adapter accepting inputs_embeds."""

    def __init__(self) -> None:
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, LM_HIDDEN)

    @property
    def capabilities(self) -> LanguageModelCapabilities:
        return LanguageModelCapabilities(
            model_id="dummy-lm",
            accepts_inputs_embeds=True,
            native_visual_connector=False,
            max_text_tokens=128,
            max_visual_tokens=64,
        )

    def tokenize(self, conversations: list[Any]) -> TokenizedText:
        ids = torch.randint(0, VOCAB_SIZE, (len(conversations), 6))
        return TokenizedText(input_ids=ids, attention_mask=torch.ones_like(ids))

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward_with_visual_tokens(
        self,
        text: TokenizedText,
        visual_tokens: ProjectedVisualTokens | None,
        labels: torch.Tensor | None,
    ) -> LanguageOutput:
        b, seq_len = text.input_ids.shape
        n_visual = visual_tokens.tokens.shape[1] if visual_tokens is not None else 0
        logits = torch.randn(b, seq_len + n_visual, VOCAB_SIZE)
        loss = logits.sum() * 0.0 if labels is not None else None
        return LanguageOutput(logits=logits, loss=loss)

    def generate(
        self,
        text: TokenizedText,
        visual_tokens: ProjectedVisualTokens | None,
        generation_config: GenerationConfig,
    ) -> GeneratedText:
        b = text.input_ids.shape[0]
        return GeneratedText(
            texts=tuple("dummy answer" for _ in range(b)),
            token_ids=torch.zeros(b, generation_config.max_new_tokens, dtype=torch.int64),
        )


class TextOnlyLanguageModelAdapter(DummyLanguageModelAdapter):
    """Adapter without any visual pathway; visual input must fail loudly."""

    @property
    def capabilities(self) -> LanguageModelCapabilities:
        return LanguageModelCapabilities(
            model_id="text-only-lm",
            accepts_inputs_embeds=False,
            native_visual_connector=False,
        )

    def forward_with_visual_tokens(
        self,
        text: TokenizedText,
        visual_tokens: ProjectedVisualTokens | None,
        labels: torch.Tensor | None,
    ) -> LanguageOutput:
        if visual_tokens is not None:
            raise UnsupportedCapabilityError(
                "text-only-lm has neither inputs_embeds support nor a native visual connector"
            )
        return super().forward_with_visual_tokens(text, None, labels)


class DummyTaskModule:
    """Protocol-conformant binary-classification task module."""

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    @property
    def task_type(self) -> TaskType:
        return TaskType.BINARY_CLASSIFICATION

    @property
    def supported_modalities(self) -> tuple[Modality, ...]:
        return (Modality.XRAY_2D, Modality.CT_2D_SLICE, Modality.MRI_2D_SLICE)

    def check_supported(self, modality: Modality) -> None:
        from medfm.core import UnsupportedModalityError

        if modality not in self.supported_modalities:
            raise UnsupportedModalityError(f"dummy task does not support {modality}")

    def compute_loss(self, model_output: EncoderOutput, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        assert model_output.pooled_embedding is not None and batch.labels is not None
        logits = model_output.pooled_embedding.sum(dim=-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels.float())
        return LossOutput(
            total=loss,
            components={"bce": loss.detach()},
            sample_count=int(batch.labels.shape[0]),
        )

    def reset_metrics(self) -> None:
        self._correct = 0
        self._total = 0

    def update_metrics(self, model_output: EncoderOutput, batch: MedicalBatch) -> None:
        assert model_output.pooled_embedding is not None and batch.labels is not None
        predictions = (model_output.pooled_embedding.sum(dim=-1) > 0).long()
        self._correct += int((predictions == batch.labels.long()).sum())
        self._total += int(batch.labels.shape[0])

    def compute_metrics(self) -> dict[str, float]:
        return {"accuracy": self._correct / self._total if self._total else 0.0}
