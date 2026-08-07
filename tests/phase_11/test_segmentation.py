from __future__ import annotations

import torch

from medfm.core.encoder import EncoderOutput
from medfm.models.decoders import (
    FPNDecoder2D,
    FPNDecoder3D,
    LanguageConditionedMaskDecoder,
    NativeModelDecoderWrapper,
    PromptableMaskDecoder,
    TransformerMaskDecoder,
    UNetDecoder2D,
    UNetDecoder3D,
)
from medfm.tasks.losses import DiceBCELoss, DiceCELoss, TverskyLoss
from medfm.tasks.segmentation import SegmentationTask
from phase_11.conftest import make_batch_2d


def test_unet_and_fpn_share_2d_3d_shapes(encoder_output_2d: EncoderOutput, encoder_output_3d: EncoderOutput) -> None:
    unet2d = UNetDecoder2D((4, 8), num_classes=2, hidden_channels=4)
    unet3d = UNetDecoder3D((4, 8), num_classes=2, hidden_channels=4)
    fpn2d = FPNDecoder2D((4, 8), num_classes=2, pyramid_channels=4)
    fpn3d = FPNDecoder3D((4, 8), num_classes=2, pyramid_channels=4)
    assert unet2d(encoder_output_2d).logits.shape == (2, 2, 8, 8)
    assert fpn2d(encoder_output_2d).logits.shape == (2, 2, 8, 8)
    assert unet3d(encoder_output_3d).logits.shape == (2, 2, 4, 8, 8)
    assert fpn3d(encoder_output_3d).logits.shape == (2, 2, 4, 8, 8)


def test_segmentation_losses_handle_empty_and_full_targets() -> None:
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    empty = torch.zeros(2, 1, 8, 8)
    full = torch.ones(2, 1, 8, 8)
    total = DiceBCELoss()(logits, empty) + DiceBCELoss()(logits, full) + TverskyLoss()(logits, full)
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    multi_logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    value = DiceCELoss()(multi_logits, labels)
    value.backward()
    assert torch.isfinite(value) and multi_logits.grad is not None


def test_deep_supervision_and_task_backward(encoder_output_2d: EncoderOutput) -> None:
    decoder = UNetDecoder2D((4, 8), num_classes=1, hidden_channels=4, deep_supervision=True)
    task = SegmentationTask(decoder, binary=True, deep_supervision_weights=(0.25, 0.75))
    batch = make_batch_2d(segmentation=torch.zeros(2, 1, 8, 8))
    loss = task.compute_loss(encoder_output_2d, batch)
    assert loss.sample_count == 2 and torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in decoder.parameters())


def test_transformer_prompt_and_native_decoder_contract() -> None:
    features = torch.randn(2, 4, 4, 4, requires_grad=True)
    queries = torch.randn(2, 2, 6, requires_grad=True)
    decoder = TransformerMaskDecoder(4, 6, mask_dim=8, num_masks=2)
    output = decoder(features, queries, query_mask=torch.tensor([[True, False], [True, True]]))
    assert output.logits.shape == (2, 2, 4, 4)
    output.logits.sum().backward()
    assert features.grad is not None and queries.grad is not None
    prompt = PromptableMaskDecoder(4, 6, mask_dim=8, num_masks=2)
    assert prompt(features.detach(), queries.detach()).logits.shape == (2, 2, 4, 4)
    native = NativeModelDecoderWrapper(lambda value: {"logits": value, "opaque": "preserved"})
    native_output = native(features.detach())
    assert native_output.native_outputs["opaque"] == "preserved"


def test_language_conditioned_masks_use_visual_spatial_decoder() -> None:
    visual = torch.randn(2, 4, 3, 5, requires_grad=True)
    text = torch.randn(2, 4, 6, requires_grad=True)
    decoder = LanguageConditionedMaskDecoder(4, 6, hidden_dim=8)
    output = decoder(visual, text, text_mask=torch.tensor([[True, True, False, False], [False, False, False, False]]))
    assert output.logits.shape == (2, 1, 3, 5)
    assert torch.allclose(output.logits[1], torch.zeros_like(output.logits[1]))
    output.logits[0].sum().backward()
    assert visual.grad is not None and text.grad is not None


def test_3d_language_conditioned_decoder() -> None:
    visual = torch.randn(2, 4, 2, 3, 3)
    text = torch.randn(2, 2, 6)
    output = LanguageConditionedMaskDecoder(4, 6, hidden_dim=8)(visual, text)
    assert output.logits.shape == (2, 1, 2, 3, 3)


def test_valid_count_padding_mask_is_exposed(encoder_output_2d: EncoderOutput) -> None:
    decoder = UNetDecoder2D((4, 8), num_classes=1, hidden_channels=4)
    batch = make_batch_2d(segmentation=torch.zeros(2, 1, 8, 8))
    batch = batch.__class__(
        modality=batch.modality,
        sample_ids=batch.sample_ids,
        pixel_values=batch.pixel_values,
        task_targets={"segmentation": batch.task_targets["segmentation"], "sample_mask": torch.tensor([True, False])},
    )
    loss = SegmentationTask(decoder, binary=True).compute_loss(encoder_output_2d, batch)
    assert loss.sample_count == 1
    assert "valid_count" in loss.diagnostics
