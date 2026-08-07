"""Phase 04 smoke test: end-to-end transform pipelines across modalities.

Chains reader-style payloads through deterministic canonicalization, the
cache boundary, stochastic augmentation, spec validation, inversion, and
collation — the phase's exit criteria exercised as one flow per modality.
"""

import pytest
import torch

from medfm.core.enums import Modality
from medfm.core.sample import SpatialMetadata
from medfm.data.collators import ClassificationCollator
from medfm.data.transforms.base import TransformContext, TransformData, invert_history
from medfm.data.transforms.ct import ClipHU, ToHounsfieldUnits, WindowChannels
from medfm.data.transforms.mri import ForegroundZScoreNormalize, SequenceResolver, stack_sequences
from medfm.data.transforms.pipeline import TransformPipeline
from medfm.data.transforms.radiology2d import (
    DecodeGrayscale,
    LetterboxResize,
    NormalizeImage,
    RandomFlip2D,
    RandomGaussianNoise,
    RescaleIntensity,
    ToChannels,
)
from medfm.data.transforms.spatial3d import (
    CanonicalizeOrientation,
    ForegroundCrop3D,
    ResampleToSpacing,
)
from medfm.data.transforms.specs import NormalizationSpec, PreprocessSpec
from medfm.data.transforms.timing import PreprocessTimer, time_host_preprocessing


def _spatial3d(shape: tuple[int, int, int], orientation: str = "LPS") -> SpatialMetadata:
    affine = torch.diag(torch.as_tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    return SpatialMetadata(
        original_shape=shape,
        current_shape=shape,
        affine=affine,
        spacing_mm=(1.5, 1.5, 3.0),
        orientation=orientation,
        anatomical_axes=tuple(orientation),
    )


# ---------------------------------------------------------------------------
# 2D radiology end-to-end
# ---------------------------------------------------------------------------


def _xray_pipeline() -> TransformPipeline:
    spec = PreprocessSpec(
        model_id="dummy-xray-v1",
        spatial_shape=(224, 224),
        channels=3,
        value_range=None,
        normalization=NormalizationSpec(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
    )
    return TransformPipeline(
        deterministic=[
            DecodeGrayscale("MONOCHROME1"),
            RescaleIntensity(out_range=(0.0, 1.0)),
            LetterboxResize((224, 224)),
            ToChannels(3),
            NormalizeImage(NormalizationSpec(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))),
        ],
        stochastic=[RandomFlip2D(allow_horizontal=True, p=0.5), RandomGaussianNoise(std_range=(0.0, 0.02))],
        spec=spec,
        name="xray2d",
    )


def test_end_to_end_xray_pipeline_spec_conformance_and_seeding():
    pipeline = _xray_pipeline()
    image = torch.linspace(0, 4095, 200 * 300).reshape(1, 200, 300)  # MONOCHROME1-style raw
    data = TransformData(image=image, metadata={"view_position": "PA", "longitudinal_index": 2})

    ctx = TransformContext.for_sample(base_seed=7, epoch=0, worker_id=0, sample_key="study-a")
    out = pipeline(data, ctx)
    pipeline.spec.validate(out.image)  # adapter receives exactly its declared format
    assert out.metadata["view_position"] == "PA" and out.metadata["longitudinal_index"] == 2

    # Same seed -> identical augmentation; different worker -> different stream.
    again = pipeline(TransformData(image=image.clone()), TransformContext.for_sample(7, 0, 0, "study-a"))
    assert torch.equal(out.image, again.image)
    other_worker = pipeline(TransformData(image=image.clone()), TransformContext.for_sample(7, 0, 1, "study-a"))
    assert not torch.equal(out.image, other_worker.image)

    # The deterministic stage is seed-independent and cache-safe.
    det_a = pipeline.run_deterministic(TransformData(image=image.clone()))
    det_b = pipeline.run_deterministic(TransformData(image=image.clone()))
    assert torch.equal(det_a.image, det_b.image)


def test_stochastic_stage_never_contaminates_cache_keys():
    pipeline = _xray_pipeline()
    # Cache-key identity covers the deterministic stage + spec only; the
    # stochastic stage and its draws are excluded by construction.
    assert pipeline.deterministic_config_hash() == _xray_pipeline().deterministic_config_hash()
    det_config = pipeline.deterministic_config_dict()
    assert det_config["stochastic"] if "stochastic" in det_config else True
    assert "stochastic" not in det_config or not det_config["stochastic"]
    assert len(det_config["deterministic"]) == len(pipeline.deterministic_transforms)


# ---------------------------------------------------------------------------
# CT end-to-end with inversion in original physical coordinates
# ---------------------------------------------------------------------------


def test_end_to_end_ct_chain_with_inversion_and_spec():
    shape = (16, 24, 32)
    raw = torch.rand(1, *shape) * 2000.0
    mask = torch.zeros(1, *shape, dtype=torch.int64)
    mask[0, 6:10, 10:16, 12:24] = 1
    data = TransformData(image=raw.clone(), targets={"mask": mask.clone()}, spatial=_spatial3d(shape))
    spec = PreprocessSpec(model_id="dummy-ct-v1", spatial_shape=(8, 12, 16), channels=2, value_range=(0.0, 1.0))
    pipeline = TransformPipeline(
        deterministic=[
            ToHounsfieldUnits(slope=1.0, intercept=-1024.0),
            CanonicalizeOrientation(target="RAS"),
            ResampleToSpacing(spacing_mm=(3.0, 3.0, 6.0)),
            ClipHU(-1024.0, 1500.0),
            WindowChannels(windows=((40.0, 400.0), (-600.0, 1500.0))),
        ],
        spec=spec,
        name="ct3d",
    )
    out = pipeline(data)
    assert out.image.shape == spec.expected_tensor_shape()
    assert out.spatial is not None and out.spatial.orientation == "RAS"

    # Invert the mask back to original physical coordinates and compare.
    restored_mask = invert_history(out.history, out.targets["mask"], mode="label")
    assert tuple(restored_mask.shape) == tuple(mask.shape)
    assert set(torch.unique(restored_mask).tolist()) <= {0, 1}

    # Foreground crop inversion is exact for image and mask (fresh payload:
    # ``data`` was mutated by the pipeline above).
    fresh = TransformData(image=raw.clone(), targets={"mask": mask.clone()}, spatial=_spatial3d(shape))
    cropped = ForegroundCrop3D(margin=1, threshold=500.0)(fresh, None)
    restored_image = invert_history(cropped.history, cropped.image)
    assert torch.equal(restored_image, raw)


# ---------------------------------------------------------------------------
# MRI end-to-end: multi-sequence stack with missing-sequence mask
# ---------------------------------------------------------------------------


def test_end_to_end_mri_multisequence_with_missing_mask():
    resolver = SequenceResolver({"T1CE": ("T1C",), "T2": ("T2W",), "FLAIR": ()})
    shape = (4, 8, 8)

    def seq(name: str, fill: float) -> TransformData:
        return TransformData(
            image=torch.full((1, *shape), fill),
            spatial=_spatial3d(shape, orientation="RAS"),
            metadata={"sequence": name},
        )

    stacked = stack_sequences(
        [seq("t1c", 5.0), seq("T2W", 9.0)], ("T1CE", "T2", "FLAIR"), resolver, allowed_missing=("FLAIR",)
    )
    assert stacked.targets["sequence_mask"].tolist() == [True, True, False]
    spec = PreprocessSpec(model_id="dummy-mri-v1", spatial_shape=shape, channels=3)
    pipeline = TransformPipeline(deterministic=[ForegroundZScoreNormalize()], spec=spec, name="mri3d")
    out = pipeline(stacked)
    assert out.image.shape == (3, *shape)
    # The missing FLAIR channel is all-zero and survives normalization untouched.
    assert float(out.image[2].abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# Collation + independent host/device timing
# ---------------------------------------------------------------------------


def test_end_to_end_collate_and_time_host_vs_device_cpu():
    pipeline = _xray_pipeline()
    collator = ClassificationCollator(Modality.XRAY_2D)
    timer = PreprocessTimer(device="cpu")

    def produce() -> int:
        examples = []
        for index in range(4):
            image = torch.rand(1, 200, 300) * 4095.0
            ctx = TransformContext.for_sample(11, 0, 0, f"sample-{index}")
            out = pipeline(TransformData(image=image), ctx)
            examples.append(
                {
                    "sample_id": f"sample-{index}",
                    "modality": Modality.XRAY_2D,
                    "image": out.image,
                    "label": torch.tensor(index % 2),
                }
            )
        with timer.time_host(items=len(examples)):
            batch = collator(examples)
        timer.record_device_wait(batch)
        return len(examples)

    host = time_host_preprocessing(produce, repeats=2)
    report = timer.report()
    assert host.items == 8
    assert report.host.items == 8 and report.host.seconds >= 0.0
    assert report.device_wait.items == 2 and report.device_wait.seconds >= 0.0
    assert report.backend == "cpu"
    summary = report.to_dict()
    assert summary["device"] == "cpu" and "host_per_item_seconds" in summary


@pytest.mark.gpu
def test_device_wait_measured_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not present")
    collator = ClassificationCollator(Modality.XRAY_2D)
    batch = collator([_xray_classification_example(i) for i in range(2)])
    timer = PreprocessTimer(device="cuda")
    elapsed = timer.record_device_wait(batch)
    assert elapsed >= 0.0
    assert timer.report().backend == "cuda"


@pytest.mark.tpu
def test_device_wait_measured_on_tpu():
    xm = pytest.importorskip("torch_xla.core.xla_model")
    device = str(xm.xla_device())
    collator = ClassificationCollator(Modality.XRAY_2D)
    batch = collator([_xray_classification_example(i) for i in range(2)])
    timer = PreprocessTimer(device=device)
    elapsed = timer.record_device_wait(batch)
    assert elapsed >= 0.0
    assert timer.report().backend == "xla"


def _xray_classification_example(index: int) -> dict:
    return {
        "sample_id": f"sample-{index}",
        "modality": Modality.XRAY_2D,
        "image": torch.rand(3, 224, 224),
        "label": torch.tensor(index % 2),
    }
