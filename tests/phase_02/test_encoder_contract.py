"""Visual encoder contract: protocol conformance, output semantics, refusals."""

import pytest
import torch
from contract_fixtures import DummyVisualEncoder, PoolingOnlyEncoder, make_batch

from medfm.core import (
    CoordinateSystem,
    EncoderOutput,
    Modality,
    OutputSpec,
    PreprocessSpec,
    ShapeContractError,
    UnsupportedCapabilityError,
    UnsupportedModalityError,
    VisualEncoder,
)


def test_dummy_encoder_conforms_to_protocol():
    assert isinstance(DummyVisualEncoder(), VisualEncoder)
    assert isinstance(PoolingOnlyEncoder(), VisualEncoder)


def test_dummy_encoder_output_semantics():
    encoder = DummyVisualEncoder()
    batch = make_batch(Modality.XRAY_2D, batch_size=2)
    output = encoder.encode(batch)
    output.check_against(OutputSpec(pooled=True, spatial_tokens=True, token_coordinates=True))
    b, n, dv = output.spatial_tokens.shape
    assert output.pooled_embedding.shape == (b, dv)
    assert output.token_mask.shape == (b, n)
    assert output.token_coordinates.shape == (b, n, 2)
    assert output.token_coordinate_system is CoordinateSystem.NORMALIZED_IMAGE
    # Normalized coordinates stay in [0, 1].
    assert 0.0 <= float(output.token_coordinates.min()) <= float(output.token_coordinates.max()) <= 1.0


def test_unsupported_modality_fails():
    encoder = DummyVisualEncoder()
    with pytest.raises(UnsupportedModalityError, match="CT_3D"):
        encoder.capabilities.require_modality(Modality.CT_3D)


def test_spatial_request_on_pooling_only_encoder_fails_not_fabricates():
    capabilities = PoolingOnlyEncoder().capabilities
    with pytest.raises(UnsupportedCapabilityError, match="silently pool"):
        OutputSpec(spatial_tokens=True).check_supported(capabilities)


def test_output_spec_check_against_missing_output():
    output = EncoderOutput(pooled_embedding=torch.zeros(2, 4))
    with pytest.raises(UnsupportedCapabilityError, match="spatial_tokens"):
        output.check_against(OutputSpec(pooled=True, spatial_tokens=True))


def test_encoder_output_shape_validation():
    with pytest.raises(ShapeContractError, match="token_mask"):
        EncoderOutput(spatial_tokens=torch.randn(2, 5, 6), token_mask=torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ShapeContractError, match="token_coordinate_system"):
        EncoderOutput(spatial_tokens=torch.randn(2, 5, 6), token_coordinates=torch.randn(2, 5, 2))
    with pytest.raises(ShapeContractError, match="native_outputs_kind"):
        EncoderOutput(pooled_embedding=torch.zeros(2, 4), native_outputs={"hidden": None})


def test_preprocess_spec_validation():
    PreprocessSpec(image_size=(16, 16), channels=1, mean=(0.5,), std=(0.25,))
    with pytest.raises(ShapeContractError):
        PreprocessSpec(image_size=(16,), channels=1)
    with pytest.raises(ShapeContractError, match="per channel"):
        PreprocessSpec(image_size=(16, 16), channels=3, mean=(0.5,))
