"""External-encoder vision-to-language bridges (Phase 09)."""

from medfm.core.language import ProjectedVisualTokens
from medfm.models.bridges.base import (
    BridgeCapabilityError,
    BridgeError,
    LinearBridge,
    LinearVisionLanguageBridge,
    MLPBridge,
    MLPVisionLanguageBridge,
    VisionLanguageBridge,
)
from medfm.models.bridges.coordinates import (
    CoordinateAwareBridge,
    CoordinateEncoder,
    CoordinateEncoder2D,
    CoordinateEncoder3D,
    CoordinateEncoderWSI,
    ImageCoordinateEncoder,
    SlideCoordinateEncoder,
    ThreeDCoordinateEncoder,
    TwoDCoordinateEncoder,
    VolumeCoordinateEncoder,
    WSICoordinateEncoder,
)
from medfm.models.bridges.placement import (
    IGNORE_INDEX,
    TokenPlacementConfig,
    VisualBoundaryEmbeddings,
    VisualTokenPlacementAdapter,
    mask_causal_labels,
    place_visual_tokens,
)
from medfm.models.bridges.resampler import PerceiverBridge, PerceiverResamplerBridge
from medfm.models.bridges.training import (
    StageConfig,
    TrainableModuleDeclaration,
    TrainingStage,
    apply_stage_freeze,
    stage_config,
    trainable_parameter_names,
)

PerceiverResampler = PerceiverResamplerBridge

__all__ = [
    "BridgeCapabilityError",
    "BridgeError",
    "CoordinateAwareBridge",
    "CoordinateEncoder",
    "CoordinateEncoder2D",
    "CoordinateEncoder3D",
    "CoordinateEncoderWSI",
    "IGNORE_INDEX",
    "ImageCoordinateEncoder",
    "LinearBridge",
    "LinearVisionLanguageBridge",
    "MLPBridge",
    "MLPVisionLanguageBridge",
    "PerceiverBridge",
    "PerceiverResampler",
    "PerceiverResamplerBridge",
    "ProjectedVisualTokens",
    "SlideCoordinateEncoder",
    "StageConfig",
    "ThreeDCoordinateEncoder",
    "TokenPlacementConfig",
    "TrainableModuleDeclaration",
    "TrainingStage",
    "TwoDCoordinateEncoder",
    "VisionLanguageBridge",
    "VisualBoundaryEmbeddings",
    "VisualTokenPlacementAdapter",
    "VolumeCoordinateEncoder",
    "WSICoordinateEncoder",
    "apply_stage_freeze",
    "mask_causal_labels",
    "place_visual_tokens",
    "stage_config",
    "trainable_parameter_names",
]
