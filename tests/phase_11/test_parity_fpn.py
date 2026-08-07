from __future__ import annotations

from collections import OrderedDict

import torch
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


def test_torchvision_fpn_top_down_kernel_does_not_match_decoder_contract() -> None:
    """Pin the measured mismatch before considering a 2D FPN transfer.

    The decoder's top-down path uses bilinear interpolation with
    ``align_corners=False``.  Torchvision's FPN uses nearest-neighbor
    interpolation, so even identity lateral/smoothing kernels diverge on a
    non-constant feature map.
    """

    candidate = FeaturePyramidNetwork([1, 1], 1).eval()
    with torch.no_grad():
        for block in (*candidate.inner_blocks, *candidate.layer_blocks):
            convolution = block[0]
            convolution.weight.zero_()
            convolution.bias.zero_()
            if convolution.kernel_size == (3, 3):
                convolution.weight[..., 1, 1] = 1
            else:
                convolution.weight[..., 0, 0] = 1

    features = OrderedDict(
        (
            ("low", torch.zeros(1, 1, 4, 4)),
            ("high", torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])),
        )
    )
    candidate_low = candidate(features)["low"]
    decoder_top_down = torch.nn.functional.interpolate(
        features["high"], size=(4, 4), mode="bilinear", align_corners=False
    )

    drift = (candidate_low - decoder_top_down).abs().max().item()
    assert drift == 0.75
    assert not torch.allclose(candidate_low, decoder_top_down, atol=1e-6, rtol=1e-6)
