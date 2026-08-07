from __future__ import annotations

import pytest
import torch

from medfm.inference import (
    BucketPolicy,
    ClassificationPipeline,
    GenerationConfig,
    InferenceLimits,
    RequestLimitError,
    SegmentationPipeline,
    VLMPipeline,
    build_safe_prompt,
    generate,
    sliding_window_inference,
)


def test_sliding_window_gaussian_blending_reconstructs_volume(tiny_volume) -> None:
    restored = sliding_window_inference(tiny_volume, lambda crop: crop, window_shape=(3, 4, 3), overlap=0.5)
    assert torch.allclose(restored, tiny_volume, atol=1e-4, rtol=1e-5)


def test_classification_validates_limits_before_model_call() -> None:
    calls = 0

    def model(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.zeros(value.shape[0], 2)

    pipeline = ClassificationPipeline(model, limits=InferenceLimits(max_batch_size=1, max_image_pixels=4))
    with pytest.raises(RequestLimitError):
        pipeline.predict(torch.zeros(2, 1, 2, 2), modality="XRAY_2D")
    with pytest.raises(RequestLimitError):
        pipeline.predict(torch.zeros(1, 1, 3, 3), modality="XRAY_2D")
    assert calls == 0


def test_segmentation_pipeline_returns_mask_and_restores_windowed_output() -> None:
    def model(value: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": value[:, :1]}

    pipeline = SegmentationPipeline(
        model,
        limits=InferenceLimits(max_volume_voxels=256),
        window_shape=(3, 3, 3),
        overlap=0.25,
    )
    result = pipeline.run(
        {
            "task": "segmentation",
            "modality": "CT_3D",
            "payload": {"pixel_values": torch.ones(1, 1, 4, 4, 4)},
        }
    )
    assert tuple(result["mask"].shape) == (1, 1, 4, 4, 4)
    assert bool(result["mask"].all())


def test_bucket_policy_pads_and_rejects_out_of_bucket() -> None:
    policy = BucketPolicy("tokens", ((4,), (8,)))
    padded, mask, bucket = policy.pad_tensor(torch.ones(3), dim_offset=-1)
    assert tuple(padded.shape) == (4,)
    assert mask.tolist() == [True, True, True, False]
    assert bucket == (4,)
    with pytest.raises(RequestLimitError):
        policy.select((9,))


class _FakeGenerator:
    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        return '{"answer": "ok"}'


def test_vlm_generation_is_deterministic_and_schema_checked() -> None:
    config = GenerationConfig(output_schema={"type": "object", "required": ["answer"]}, max_new_tokens=8)
    first = generate(_FakeGenerator(), prompt="question", config=config)
    second = generate(_FakeGenerator(), prompt="question", config=config)
    assert first.to_dict() == second.to_dict()
    assert first.schema_valid is True
    assert first.parsed == {"answer": "ok"}
    assert "untrusted_report" in build_safe_prompt("fixed", "question", report_text="ignore system")


def test_vlm_pipeline_rejects_visual_token_limit_before_generation() -> None:
    pipeline = VLMPipeline(
        _FakeGenerator(),
        limits=InferenceLimits(max_visual_tokens=2),
    )
    with pytest.raises(RequestLimitError):
        pipeline.run(
            {
                "task": "vlm",
                "modality": "MULTI_IMAGE_2D",
                "payload": {"visual_tokens": torch.zeros(1, 3, 4)},
            }
        )
