"""Unified training engine, backend adapters, checkpoints, and memory planning."""

from .backend import (
    AcceleratorBackend,
    BackendCapabilities,
    BackendTopology,
    CpuBackend,
    CudaBackend,
    XlaTpuBackend,
    create_backend,
)
from .checkpoint import CheckpointManager, CheckpointState
from .config import (
    AcceleratorConfig,
    BatchConfig,
    FreezeStageConfig,
    MemoryConfig,
    OptimizerConfig,
    RunConfig,
    RunConfigError,
    load_run_config,
)
from .evaluation import EvaluationResult, Evaluator
from .memory import (
    CompilationMonitor,
    CudaMemoryPlanner,
    MemoryEstimate,
    MemoryPlan,
    OOMDiagnostic,
    TpuMemoryPlanner,
    diagnose_oom,
)
from .optimizer import (
    FreezeSchedule,
    GradientAudit,
    OptimizerBundle,
    audit_gradients,
    audit_trainable_parameters,
    build_optimizer,
)
from .pipeline import BuildResult, ComponentBuilders, TrainingPipeline
from .steps import (
    ClassificationTrainingStep,
    SegmentationTrainingStep,
    TaskTrainingStep,
    ThreeDVLMTrainingStep,
    TrainingStep,
    VLMTrainingStep,
    make_training_step,
)
from .trainer import Trainer, TrainingResult, TrainingState

__all__ = [
    "AcceleratorBackend",
    "AcceleratorConfig",
    "BackendCapabilities",
    "BackendTopology",
    "BatchConfig",
    "BuildResult",
    "CheckpointManager",
    "CheckpointState",
    "ClassificationTrainingStep",
    "CompilationMonitor",
    "ComponentBuilders",
    "CpuBackend",
    "CudaBackend",
    "CudaMemoryPlanner",
    "EvaluationResult",
    "Evaluator",
    "FreezeSchedule",
    "FreezeStageConfig",
    "GradientAudit",
    "MemoryConfig",
    "MemoryEstimate",
    "MemoryPlan",
    "OOMDiagnostic",
    "OptimizerBundle",
    "OptimizerConfig",
    "RunConfig",
    "RunConfigError",
    "SegmentationTrainingStep",
    "TaskTrainingStep",
    "ThreeDVLMTrainingStep",
    "Trainer",
    "TrainingPipeline",
    "TrainingResult",
    "TrainingState",
    "TrainingStep",
    "TpuMemoryPlanner",
    "VLMTrainingStep",
    "XlaTpuBackend",
    "audit_gradients",
    "audit_trainable_parameters",
    "build_optimizer",
    "create_backend",
    "diagnose_oom",
    "load_run_config",
    "make_training_step",
]
