# Unified Medical Foundation-Model Framework

## Agentic, phase-gated implementation plan

No blocking clarification is required. I am proceeding with these assumptions:

* One Linux workstation with one NVIDIA GPU and **48GB VRAM**.
* PyTorch, MONAI and Hugging Face form the main runtime.
* Training is predominantly **LoRA, QLoRA, frozen-encoder adaptation, projector training and decoder/head training**.
* The framework must support:

  * 2D radiology and pathology patches.
  * 3D CT and MRI.
  * Whole-slide pathology.
  * Classification.
  * Segmentation.
  * Retrieval and contrastive learning.
  * 2D VLM instruction tuning.
  * Native 3D VLM instruction tuning.
  * Slice-sequence VLM training.
  * Language-conditioned segmentation.
* Full pretraining of billion-parameter foundation models is outside the initial scope.
* Clinical data is de-identified before entering the training environment.
* Each model remains modality-specific internally, but all models implement a shared framework contract.

The correct abstraction is:

```text
One framework
    ├── Multiple image readers
    ├── Multiple modality preprocessors
    ├── Multiple visual foundation-model adapters
    ├── Multiple language-model adapters
    ├── Shared vision-to-language bridges
    ├── Shared PEFT engine
    ├── Shared task heads
    ├── Shared trainer and evaluator
    └── Shared checkpoint and deployment format
```

MONAI should provide the medical-imaging foundation because it supports medical-specific transforms, 2D and 3D architectures, sliding-window inference and reproducible model bundles. Hugging Face PEFT, Transformers, TRL, Accelerate and bitsandbytes should provide the LLM/VLM adaptation layer. ([MONAI][1])

---

# Agentic execution protocol

Before implementing individual components, establish a strict protocol for coding agents.

## Every phase must have

* A single scoped objective.
* Explicit allowed files.
* Explicit forbidden files.
* Required implementation artifacts.
* Unit tests.
* Integration tests where applicable.
* One smoke command.
* One acceptance command.
* A phase report.
* A machine-readable completion manifest.
* No unresolved test failures.
* No silently skipped tests.
* No untracked architectural decisions.

## Required phase files

Create:

```text
agent/
├── README.md
├── phase_template.md
├── acceptance_schema.json
├── prompts/
│   ├── implement_phase.md
│   ├── review_phase.md
│   ├── test_phase.md
│   └── repair_phase.md
└── reports/
```

Each completed phase should generate:

```text
agent/reports/phase_<NN>/
├── summary.md
├── files_changed.txt
├── commands_executed.txt
├── test_results.json
├── acceptance.json
├── unresolved_issues.md
└── next_phase_handoff.md
```

## Standard agent instruction

Use the following pattern for each phase:

```text
You are implementing Phase <N> of the medical foundation-model framework.

Read:
1. docs/architecture/*
2. agent/reports/phase_<N-1>/next_phase_handoff.md
3. The phase specification.
4. Existing tests for the affected modules.

Constraints:
- Modify only files explicitly allowed by the phase specification.
- Preserve all public interfaces unless the phase explicitly changes them.
- Do not introduce an additional framework when existing dependencies suffice.
- Do not download patient data.
- Do not place model weights in Git.
- Do not fabricate successful test results.
- Do not proceed past a failing acceptance condition.

Required completion:
- Implement the requested code.
- Add or update tests.
- Run all specified commands.
- Write the phase report.
- Record unresolved issues.
- Produce a next-phase handoff.
```

## Agent completion gate

Every phase must run something structurally equivalent to:

```bash
python -m medfm.tools.validate_phase --phase <N>
pytest tests/phase_<N> -q
python -m medfm.tools.smoke --phase <N>
```

The validation utility should verify:

* Required files exist.
* Required tests ran.
* The phase report is populated.
* No acceptance criterion is marked “unknown.”
* No model license is missing.
* No dataset lacks provenance.
* No checkpoint lacks a base-model reference and configuration hash.

---

# Dependency graph

Implement in this order:

```text
Phase 0: Requirements and governance
        ↓
Phase 1: Repository and environment
        ↓
Phase 2: Core type system and contracts
        ↓
Phase 3: Dataset manifests and ingestion
        ↓
Phase 4: Preprocessing and transforms
        ↓
Phase 5: Model registry and licensing
        ↓
Phase 6: 2D visual adapters
Phase 7: 3D visual adapters
Phase 8: Pathology and WSI adapters
        ↓
Phase 9: Language models and visual bridges
        ↓
Phase 10: LoRA/QLoRA subsystem
        ↓
Phase 11: Task heads and losses
        ↓
Phase 12: Unified training engine
        ↓
Phase 13: 2D task recipes
Phase 14: 3D task recipes
Phase 15: Pathology task recipes
        ↓
Phase 16: Evaluation and validation
        ↓
Phase 17: Inference, export and serving
        ↓
Phase 18: Hardening, CI and release
```

Phases 6, 7 and 8 can be implemented in parallel after Phase 5, but Phase 9 must not begin until each adapter family has at least one working implementation.

---

# Phase 0 — Requirements, scope and governance

## Objective

Freeze the framework’s supported modalities, tasks, model families, legal constraints and first-release acceptance criteria.

## Write

Create:

```text
docs/
├── product_requirements.md
├── supported_tasks.md
├── supported_modalities.md
├── clinical_safety_scope.md
├── data_governance.md
├── model_governance.md
├── licensing_policy.md
├── reproducibility_policy.md
└── architecture/
    ├── adr_0001_single_framework_multiple_backbones.md
    ├── adr_0002_peft_first_training.md
    ├── adr_0003_external_encoder_vlm_bridge.md
    ├── adr_0004_patient_level_splitting.md
    ├── adr_0005_native_3d_and_slice_sequence_vlm.md
    └── adr_0006_adapter_only_checkpoints.md
```

## Define supported modalities

Use canonical modality names:

```text
XRAY_2D
CT_2D_SLICE
CT_3D
MRI_2D_SLICE
MRI_3D
PATHOLOGY_TILE
PATHOLOGY_WSI
MULTI_IMAGE_2D
MULTI_SERIES_3D
TEXT_ONLY
```

## Define supported tasks

```text
BINARY_CLASSIFICATION
MULTICLASS_CLASSIFICATION
MULTILABEL_CLASSIFICATION
ORDINAL_CLASSIFICATION
IMAGE_TEXT_RETRIEVAL
TEXT_IMAGE_RETRIEVAL
SEMANTIC_SEGMENTATION
INSTANCE_SEGMENTATION
PROMPTABLE_SEGMENTATION
LANGUAGE_CONDITIONED_SEGMENTATION
BOUNDING_BOX_LOCALIZATION
VISUAL_QUESTION_ANSWERING
REPORT_GENERATION
STRUCTURED_FINDING_GENERATION
CONTRASTIVE_ALIGNMENT
MULTITASK
```

## Define the v1 model roster

### 2D

* MedSigLIP.
* RAD-DINO.
* MedGemma 1.5 native visual pathway.
* H-Optimus-0.
* Optional CONCH.

MedSigLIP provides separate medical image and text towers and is specifically intended for classification, zero-shot classification and semantic retrieval. It uses a 400M-parameter visual tower, a 400M-parameter text tower and 448×448 images. ([Google for Developers][2])

### 3D CT

* CT-FM.
* FlexiCT-3D.
* Merlin.
* M3D-CLIP/M3D-LaMed as a research integration.

CT-FM exposes a 3D CT foundation model trained for segmentation, classification and retrieval-oriented tasks. FlexiCT exposes separate 2D, 3D and 3D vision-language checkpoints with CLS and patch-token outputs. Merlin is a native 3D CT VLM with public code and PEFT dependencies. ([GitHub][3])

### 3D MRI

* Triad.
* NV-Segment-CTMR.
* Optional BrainIAC integration later.

Triad supplies 3D MRI Swin-based checkpoints for segmentation, classification and registration. NV-Segment-CTMR supplies automatic 3D CT/MRI segmentation models and supports more than 300 anatomical classes. ([GitHub][4])

### Promptable segmentation

* MedSAM2.
* NV-Segment-CTMR interactive pathways where supported.

MedSAM2 applies a SAM2-style memory architecture to 3D medical images and videos and uses prompts and prior-slice memory to produce segmentation masks. ([MedSAM2][5])

### Pathology

* H-Optimus-0.
* GigaPath-Flash.
* TITAN.
* Optional CONCH.

H-Optimus-0 is a 1.1B-parameter pathology encoder released under Apache 2.0. GigaPath uses separate tile and slide encoders, while GigaPath-Flash provides a lighter slide-level path. TITAN supplies multimodal whole-slide representations aligned with pathology reports. ([Bioptimus][6])

### Language and generative VLM

* MedGemma 1.5 4B as the primary medical generative model.
* M3D-LaMed as a native 3D research baseline.
* Optional generic Gemma/Qwen language models behind the same interface.

MedGemma 1.5 4B supports high-dimensional medical inputs through collections of CT/MRI slices and sampled WSI images; it should be treated as a slice/multi-image architecture rather than a substitute for a native volumetric encoder. ([Google for Developers][7])

## Write a license registry

For every checkpoint record:

```yaml
model_id:
provider:
repository:
weights_uri:
code_license:
weights_license:
commercial_use:
derivative_models:
redistribution:
gated_access:
accepted_terms_date:
approved_use_cases:
prohibited_use_cases:
review_owner:
review_date:
```

A model must not be loadable through the production registry until these fields are populated.

## Exit criteria

* Every modality maps to at least one backbone.
* Every requested task maps to at least one task implementation.
* Every backbone has a preliminary license record.
* Commercial and research-only models are clearly separated.
* The system explicitly states that generated outputs require task-specific clinical validation.

---

# Phase 1 — Repository, environment and reproducibility

## Objective

Create a reproducible development environment that works for CPU tests and one 48GB CUDA GPU.

## Repository structure

```text
medical-fm/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── Makefile
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.ci
│   └── compose.yaml
├── configs/
├── medfm/
│   ├── __init__.py
│   ├── cli/
│   ├── core/
│   ├── data/
│   ├── models/
│   ├── peft/
│   ├── tasks/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── registry/
│   └── tools/
├── tests/
├── scripts/
├── docs/
├── examples/
├── model_registry/
├── agent/
└── artifacts/
```

## Dependency groups

### Core

* `torch`
* `torchvision`
* `numpy`
* `scipy`
* `pandas`
* `einops`
* `safetensors`
* `pydantic`
* `pyyaml`
* `fsspec`

### Medical imaging

* `monai`
* `nibabel`
* `pydicom`
* `highdicom`
* `SimpleITK`
* `scikit-image`

pydicom handles low-level DICOM datasets and pixel data. highdicom adds higher-level interfaces for computational imaging, frame selection and derived DICOM objects. ([Pydicom][8])

### Pathology

* `openslide-python`
* `tiffslide`
* `cucim`, enabled conditionally.
* `h5py`
* `zarr`
* `pyarrow`

OpenSlide is designed to read gigapixel WSI pyramids without loading an entire slide into memory. cuCIM supplies GPU-accelerated multidimensional image I/O and processing. ([OpenSlide][9])

### Hugging Face and PEFT

* `transformers`
* `peft`
* `accelerate`
* `bitsandbytes`
* `datasets`
* `huggingface_hub`
* `trl`

PEFT supplies LoRA and related parameter-efficient methods; bitsandbytes integrates 4-bit QLoRA-style loading; Accelerate provides mixed-precision and gradient-accumulation orchestration; TRL provides supervised fine-tuning utilities. ([Hugging Face][10])

### Quality

* `pytest`
* `pytest-cov`
* `ruff`
* `mypy` or `pyright`
* `pre-commit`

### Tracking

Prefer a local-first abstraction:

```text
Tracker
├── LocalJSONTracker
├── TensorBoardTracker
├── MLflowTracker
└── OptionalWandBTracker
```

Do not make external tracking services mandatory because medical metadata may be sensitive.

## Environment commands

Write:

```bash
make install
make install-dev
make lint
make typecheck
make test
make test-gpu
make smoke
make doctor
```

`make doctor` should print:

* Python version.
* PyTorch version.
* CUDA runtime.
* GPU model.
* Available VRAM.
* BF16 availability.
* bitsandbytes availability.
* FlashAttention or SDPA availability.
* MONAI version.
* Transformers/PEFT versions.
* Free disk space.
* Model cache path.
* Dataset cache path.

## Reproducibility rules

Record in every training run:

* Git commit.
* Dirty working-tree state.
* Python lockfile hash.
* CUDA and driver versions.
* GPU model.
* Random seed.
* Dataset-manifest hash.
* Preprocessing-configuration hash.
* Base-model revision or commit SHA.
* Adapter configuration.
* Trainable-parameter count.
* Precision mode.
* Effective batch size.
* Maximum allocated VRAM.

## Exit criteria

* CPU unit tests run without model weights.
* A CUDA smoke test allocates BF16 tensors.
* A minimal LoRA module can perform one optimization step.
* A minimal MONAI 3D transform can load and crop a synthetic NIfTI volume.
* No patient data or model weights are committed to Git.

---

# Phase 2 — Core type system and model contracts

## Objective

Define a strict common contract for 2D, 3D, WSI and language data.

## Core sample schema

Create:

```python
@dataclass
class MedicalSample:
    sample_id: str
    patient_id_hash: str
    study_id_hash: str | None
    series_id_hash: str | None

    modality: Modality
    image_references: list[ImageReference]

    labels: LabelTarget | None
    segmentation: SegmentationTarget | None
    boxes: BoxTarget | None

    report: str | None
    question: str | None
    answer: str | None
    conversations: list[ConversationTurn] | None

    spatial: SpatialMetadata | None
    pathology: PathologyMetadata | None
    provenance: ProvenanceMetadata
```

## Spatial metadata

```python
@dataclass
class SpatialMetadata:
    affine: Tensor | None
    original_affine: Tensor | None
    spacing_mm: tuple[float, ...] | None
    orientation: str | None
    original_shape: tuple[int, ...]
    current_shape: tuple[int, ...]
    anatomical_axes: tuple[str, ...] | None
    slice_positions_mm: Tensor | None
    frame_of_reference_hash: str | None
```

Never discard affine, spacing or orientation data during preprocessing.

## Pathology metadata

```python
@dataclass
class PathologyMetadata:
    microns_per_pixel: float | None
    magnification: float | None
    slide_dimensions: tuple[int, int] | None
    level_dimensions: list[tuple[int, int]]
    stain: str | None
    scanner_vendor: str | None
    tile_coordinates: Tensor | None
```

## Unified batch

```python
@dataclass
class MedicalBatch:
    pixel_values: Tensor | None
    image_mask: Tensor | None
    tile_coordinates: Tensor | None
    spatial_metadata: list[SpatialMetadata | None]

    input_ids: Tensor | None
    attention_mask: Tensor | None
    labels: Tensor | None

    task_targets: dict[str, Any]
    sample_ids: list[str]
```

Supported shapes:

```text
2D image:        [B, C, H, W]
3D volume:       [B, C, D, H, W]
Multi-image:     [B, I, C, H, W]
WSI tiles:       [B, T, C, H, W]
Visual tokens:   [B, N, Dv]
Text tokens:     [B, L]
Segmentation:    [B, K, H, W] or [B, K, D, H, W]
```

Never infer modality solely from tensor rank. The `modality` field must be authoritative.

## Visual encoder contract

```python
class VisualEncoder(Protocol):
    @property
    def capabilities(self) -> EncoderCapabilities: ...

    def preprocess_spec(self) -> PreprocessSpec: ...

    def encode(
        self,
        batch: MedicalBatch,
        output_hidden_states: bool = False,
    ) -> "EncoderOutput": ...
```

```python
@dataclass
class EncoderOutput:
    pooled_embedding: Tensor | None
    spatial_tokens: Tensor | None
    feature_maps: list[Tensor] | None

    token_mask: Tensor | None
    token_coordinates: Tensor | None

    logits: Tensor | None
    native_outputs: Any | None
    auxiliary: dict[str, Any]
```

## Important semantic requirements

* `pooled_embedding` represents an image, volume or slide.
* `spatial_tokens` preserve patch-, slice-, tile- or volume-patch-level information.
* `feature_maps` are used by segmentation decoders.
* `token_coordinates` must use a documented coordinate system:

  * Normalized image coordinates.
  * Millimetres for radiology.
  * Microns or slide pixels for pathology.
* `token_mask` distinguishes real tokens from padded tokens.
* Adapters must not silently pool tokens when spatial output was requested.

## Language-model contract

```python
class LanguageModelAdapter(Protocol):
    def tokenize(self, conversations: list[Any]) -> TokenizedText: ...

    def embed_tokens(self, input_ids: Tensor) -> Tensor: ...

    def forward_with_visual_tokens(
        self,
        text: TokenizedText,
        visual_tokens: ProjectedVisualTokens,
        labels: Tensor | None,
    ) -> LanguageOutput: ...

    def generate(
        self,
        text: TokenizedText,
        visual_tokens: ProjectedVisualTokens,
        generation_config: GenerationConfig,
    ) -> GeneratedText: ...
```

## Task contract

```python
class TaskModule(Protocol):
    def compute_loss(
        self,
        model_output: Any,
        batch: MedicalBatch,
    ) -> LossOutput: ...

    def update_metrics(
        self,
        model_output: Any,
        batch: MedicalBatch,
    ) -> None: ...
```

## Exit criteria

* Synthetic 2D, 3D and WSI batches pass schema validation.
* Incorrect tensor rank and modality combinations fail clearly.
* Spatial metadata survives a round-trip through a transform pipeline.
* Every output tensor has documented shape semantics.

---

# Phase 3 — Dataset manifests, ingestion and provenance

## Objective

Create a unified dataset layer without forcing all modalities into one physical format.

## Canonical manifest

Use Parquet for primary manifests and permit JSONL for debugging.

Required columns:

```text
sample_id
patient_id_hash
study_id_hash
series_id_hash
modality
image_uri
secondary_image_uris
mask_uri
annotation_uri
report_uri
label_json
split
site_id
scanner_vendor
acquisition_date_bucket
dataset_name
dataset_version
license
provenance_uri
```

Do not put raw reports directly in a general-purpose manifest when storage policy requires restricted access. Permit encrypted or access-controlled text references.

## Patient-level split policy

Splits must be generated by:

1. Patient.
2. Then site, where external-site validation is possible.
3. Then time, for temporal validation where appropriate.

Prevent:

* Different studies from one patient crossing splits.
* WSI tiles from one slide crossing splits.
* Different slides from one case crossing splits.
* Adjacent slices from one volume crossing splits.
* Derived or resampled copies crossing splits.

## Dataset fingerprinting

Implement:

```text
medfm data fingerprint --manifest <path>
```

Produce:

* Number of patients.
* Number of studies.
* Number of samples.
* Modalities.
* Shape distributions.
* Spacing distributions.
* Intensity percentiles.
* Label prevalence.
* Missing values.
* Scanner/site distribution.
* Duplicate hashes.
* Split leakage checks.
* Report-length distribution.
* WSI magnification and MPP distribution.
* Segmentation class volume distribution.

nnU-Net’s dataset-fingerprinting concept is a useful precedent: it examines dataset properties to configure preprocessing, patch size and training behavior. Your framework should use the concept without becoming dependent on nnU-Net internals. ([GitHub][11])

## Radiology readers

Implement:

```text
DICOMSeriesReader
NiftiReader
MHAReader
NumpyVolumeReader
PngJpegReader
DICOMWebReader        # optional later
```

### DICOM requirements

* Sort slices using physical position, not filenames.
* Validate consistent orientation.
* Validate pixel spacing.
* Apply rescale slope/intercept for CT.
* Handle MONOCHROME1 inversion where required.
* Record acquisition metadata separately from training tensors.
* Detect multiframe objects.
* Detect localizer/scout series.
* Reject mixed series unless explicitly configured.
* Preserve enough metadata to map output masks back to source coordinates.
* Hash UIDs rather than exposing originals.

## Pathology readers

Implement:

```text
OpenSlideReader
CuCIMSlideReader
TiffSlideReader
PreExtractedTileReader
EmbeddingStoreReader
```

The reader contract must support:

* Pyramid levels.
* Region reads.
* Tile reads.
* MPP lookup.
* Tissue thumbnail generation.
* Slide dimensions.
* Coordinate conversion between pyramid levels.

## Dataset caching

Support three cache types:

```text
PreprocessingCache
VisualEmbeddingCache
TokenizationCache
```

Each cache key must include:

```text
source_file_hash
reader_version
preprocessing_hash
model_id
model_revision
output_layer
dtype
```

Never reuse an embedding cache after:

* Changing the visual model.
* Changing the visual adapter.
* Changing image normalization.
* Changing resolution or crop policy.
* Changing the extracted hidden layer.

## Exit criteria

* Synthetic DICOM series loads in physical order.
* NIfTI affine and spacing are preserved.
* WSI regions can be retrieved by coordinate.
* Patient leakage detection fails a deliberately corrupted split.
* Cache invalidation works after changing preprocessing.

---

# Phase 4 — Preprocessing, augmentation and collators

## Objective

Separate deterministic medical preprocessing from stochastic training augmentation.

## Pipeline structure

```text
Reader
  ↓
Deterministic canonicalization
  ↓
Modality-specific normalization
  ↓
Optional cache boundary
  ↓
Task-specific crop/sampling
  ↓
Stochastic augmentation
  ↓
Model-specific final transform
  ↓
Batch collation
```

## 2D radiology preprocessing

Support:

* DICOM grayscale decoding.
* MONOCHROME1 correction.
* Intensity scaling.
* Letterboxing or aspect-preserving resize.
* Optional lung or body-region crop.
* Single-channel and repeated three-channel output.
* View-position metadata.
* Multi-view grouping.
* Longitudinal ordering.

Do not use generic natural-image color jitter for grayscale radiology by default.

Recommended augmentations:

* Small rotations.
* Small translations.
* Mild scaling.
* Mild intensity shifts.
* Controlled Gaussian noise.
* Horizontal flipping only when clinically and task appropriate.
* No vertical flips unless explicitly validated.

## CT preprocessing

Implement:

* DICOM to calibrated Hounsfield units.
* Canonical orientation.
* Configurable target spacing.
* Configurable HU clipping.
* Single-window mode.
* Multi-window channel mode.
* Foreground or body crop.
* Patch sampling.
* Positive-region sampling for small lesions.
* Resampling with separate interpolation modes for images and masks.
* Invertible transform history.

Example window configuration:

```yaml
ct_windows:
  - name: soft_tissue
    low: -150
    high: 250
  - name: lung
    low: -1000
    high: 400
  - name: bone
    low: -500
    high: 1500
```

Do not hard-code these windows globally. Different pretrained models may expect different input conventions.

## MRI preprocessing

Implement:

* Sequence identification:

  * T1.
  * T1 post-contrast.
  * T2.
  * FLAIR.
  * DWI.
  * ADC.
  * Other configurable sequences.
* Canonical orientation.
* Spacing normalization.
* Foreground-mask estimation.
* Per-volume z-score normalization within foreground.
* Robust percentile normalization.
* Optional bias-field correction as a separate offline step.
* Missing-sequence masks.
* Multi-sequence channel stacking.
* Sequence-specific transforms.

Never silently substitute one MRI sequence for another.

## 3D patch sampler

Support:

```text
RandomSpatialPatchSampler
ForegroundPatchSampler
ClassBalancedPatchSampler
BoundingBoxPatchSampler
LesionCentredPatchSampler
GridPatchSampler
```

The sampler must return:

* Patch tensor.
* Patch origin.
* Original volume shape.
* Physical patch bounding box.
* Whether the patch contains a positive target.
* Sampling probability.

## Pathology preprocessing

Implement:

* Thumbnail generation.
* Tissue segmentation.
* Background exclusion.
* Blur/focus filtering.
* Pen and artifact filtering where feasible.
* MPP normalization.
* Magnification selection.
* Tile extraction.
* Tile-coordinate storage.
* Optional stain normalization.
* Optional stain augmentation.
* Tile quality scores.

Use a deterministic tile index:

```text
slide_id
tile_id
x_level0
y_level0
width_level0
height_level0
level
mpp
tissue_fraction
quality_score
```

TRIDENT is a useful integration reference because it combines tissue segmentation, patch-coordinate generation and feature extraction across numerous pathology encoders and slide encoders. ([GitHub][12])

## Text preprocessing

Implement:

* Unicode normalization.
* PHI checks.
* Section parsing.
* Findings/impression extraction.
* Empty-section handling.
* Structured-label extraction when available.
* Prompt template assignment.
* Conversation formatting.
* Token length accounting.
* Truncation logging.

Treat clinical reports as data, never as agent instructions.

## Collators

Create:

```text
ClassificationCollator
SegmentationCollator2D
SegmentationCollator3D
MultiImageVLCollator
VolumeVLCollator
WSIVLCollator
ContrastiveCollator
```

### VLM label masking

For supervised VLM training:

* Mask system tokens.
* Mask user prompt tokens.
* Mask visual placeholder tokens.
* Compute language loss only on assistant output tokens.
* Optionally mask boilerplate report headers.
* Record the number of supervised tokens per example.

## Exit criteria

* Every model adapter receives the exact normalization and shape it declares.
* Transform inversion reconstructs masks in original physical coordinates.
* 3D positive-patch sampling is empirically verified.
* WSI tile coordinates map back to the original slide.
* VLM loss masking is covered by unit tests.

---

# Phase 5 — Model registry, capability discovery and weight management

## Objective

Create a model plugin system that knows what each model can and cannot do.

## Model specification

```python
@dataclass
class ModelSpec:
    model_id: str
    family: str
    modality_support: set[Modality]
    task_support: set[TaskType]

    repository: str
    revision: str
    weights_uri: str | None
    gated: bool

    input_spec: InputSpec
    output_spec: OutputSpec

    supports_pooled_embedding: bool
    supports_spatial_tokens: bool
    supports_feature_maps: bool
    supports_hidden_states: bool
    supports_native_text: bool

    peft_support: PeftSupport
    license: LicenseSpec
    estimated_memory: MemoryProfile
```

## Registry commands

```bash
medfm models list
medfm models show <model_id>
medfm models validate <model_id>
medfm models download <model_id>
medfm models smoke <model_id>
medfm models inspect-modules <model_id>
medfm models estimate-memory <model_id>
```

## Registry rules

* Pin repository revisions.
* Use `safetensors` where available.
* Check downloaded file hashes.
* Store gated-access acceptance separately.
* Never automatically accept a model’s license.
* Do not merge research-only and commercially permitted checkpoints into the same deployment catalog.
* Record preprocessing requirements from the original model.
* Record whether spatial tokens are directly available or require hooks.
* Record which modules are valid LoRA targets.

## Model loading modes

```text
FULL_PRECISION
BF16
FP16
INT8_INFERENCE
INT8_TRAINING
NF4_QLORA
FROZEN_CPU_OFFLOAD
FROZEN_EMBEDDING_CACHE
```

## Exit criteria

* Registry validation rejects an incomplete license.
* Registry validation rejects missing input normalization.
* Each v1 model can perform an inference smoke test or is marked blocked with an explicit reason.
* Exact model revision is written into every run artifact.

---

# Phase 6 — 2D visual-encoder adapters

## Objective

Implement the first standardized 2D encoders.

## Required adapters

```text
MedSigLIPAdapter
RADDINOAdapter
HOptimus0Adapter
MedGemmaVisionAdapter
OptionalCONCHAdapter
GenericTimmViTAdapter
GenericHFVisionAdapter
```

## MedSigLIP adapter

Must support:

* Image embeddings.
* Text embeddings.
* Image-text similarity.
* Patch-token extraction where available.
* Frozen encoder mode.
* Vision LoRA mode.
* Contrastive tuning mode.
* Classification head attachment.
* External-VLM bridge attachment.

MedSigLIP was designed around a shared medical image-text embedding space and is appropriate for data-efficient classification and retrieval. ([Google for Developers][2])

## RAD-DINO adapter

Use primarily for:

* Chest X-ray classification.
* Dense patch features.
* Chest X-ray segmentation backbones.
* Retrieval.
* External VLM visual tokens.

## H-Optimus-0 adapter

Must support:

* Tile embedding.
* CLS token.
* Patch tokens.
* Intermediate hidden states.
* Frozen extraction.
* Vision LoRA.
* Embedding-cache generation.

Because H-Optimus-0 is large, the default mode should be:

```text
BF16 frozen visual encoder
+ trainable task head
or
BF16 frozen visual encoder
+ cached embeddings
```

Only enable LoRA after the frozen baseline is complete.

## Model-specific preprocessors

Do not place preprocessing inside the model’s `forward`. Each adapter must expose:

```python
adapter.preprocess_spec()
```

This returns:

* Required dimensions.
* Channel count.
* Mean and standard deviation.
* Pixel range.
* Crop behavior.
* Color space.
* Maximum images per sample.
* Patch size.
* Whether dynamic resolution is permitted.

## Acceptance tests

For every adapter:

* Load model.
* Run one synthetic input.
* Validate output shapes.
* Verify frozen mode has zero trainable backbone parameters.
* Add a classification head and complete one backward pass.
* Add LoRA and verify only intended modules receive gradients.
* Save and reload adapter weights.
* Produce numerically close results after reload.

---

# Phase 7 — Native 3D visual-encoder adapters

## Objective

Support volumetric CT/MRI models without collapsing the volume into independent slices.

## Required adapters

```text
CTFMAdapter
FlexiCT2DAdapter
FlexiCT3DAdapter
FlexiCTVLMAdapter
MerlinAdapter
TriadAdapter
NVSegmentCTMRAdapter
MedSAM2Adapter
M3DCLIPAdapter
M3DLaMedAdapter
GenericMONAI3DAdapter
```

## Standard native-3D output

Every native 3D encoder should attempt to expose:

```text
pooled_embedding: [B, Dv]
spatial_tokens:   [B, N3d, Dv]
feature_maps:
  [B, C1, D1, H1, W1]
  [B, C2, D2, H2, W2]
  ...
token_coordinates: [B, N3d, 3]
```

If a model does not naturally expose one of these:

* Return `None`.
* Record the limitation in `EncoderCapabilities`.
* Never fabricate a spatial map by reshaping an unrelated pooled vector.

## CT-FM adapter

Implement:

* Native checkpoint loading.
* Image embedding mode.
* Intermediate feature extraction.
* Classification attachment.
* Segmentation-decoder attachment.
* Retrieval embedding mode.
* Optional vision LoRA for transformer or linear components.

CT-FM was pretrained on a large 3D CT collection and evaluated across segmentation, triage and retrieval tasks, making it an appropriate permissively licensed general CT backbone. ([GitHub][3])

## FlexiCT adapter family

Expose three distinct registry entries:

```text
flexict_2d
flexict_3d
flexict_3d_vlm
```

The public wrappers distinguish 2D slice inputs, 3D volume inputs and volume-text inputs, and expose CLS and patch tokens for the image encoders. ([GitHub][13])

Do not disguise these as one implementation with ambiguous behavior. They may share internal utility code but should have separate capabilities.

## Merlin adapter

Support:

* Image embeddings.
* Image-text contrastive outputs.
* Phenotype predictions where checkpoint-compatible.
* Report-generation functionality where available.
* External head attachment.
* Reuse of Merlin preprocessing.
* PEFT loading.

Merlin’s public package already depends on MONAI, Transformers, PEFT and Accelerate, which makes it a useful reference integration for this framework. ([GitHub][14])

## Triad adapter

Support:

* MRI sequence channels.
* Swin intermediate features.
* MAE and SimMIM checkpoints as distinct registry variants.
* Segmentation decoder attachment.
* Classification pooling.
* 3D token extraction.

## NV-Segment-CTMR adapter

Treat this as a native task model first:

```text
Input -> native segmentation output
```

Then add optional support for:

* Feature extraction.
* Existing decoder fine-tuning.
* Prompt or interactive branch if available in the selected bundle.
* Adapter injection only after architecture inspection.

Because NVIDIA distributes these models through a MONAI bundle-style structure, preserve the bundle’s metadata and preprocessing rather than rebuilding it from memory. ([GitHub][15])

## MedSAM2 adapter

Expose:

```text
initialize_volume
encode_frame_or_slice
apply_prompt
update_memory
decode_mask
```

Keep its sequential-volume semantics separate from native 3D token encoders.

## M3D adapters

* `M3DCLIPAdapter` for 3D image-text alignment and retrieval.
* `M3DLaMedAdapter` for end-to-end multimodal generation experiments.

M3D-LaMed supports 3D retrieval, report generation, VQA, localization and segmentation-oriented tasks, but its larger language component should be operated with QLoRA on a 48GB GPU. ([GitHub][16])

## Acceptance tests

* CT and MRI synthetic volumes are not transposed incorrectly.
* Spacing is present in metadata.
* Patch-token coordinate grids match token order.
* A cropped volume completes forward/backward.
* Sliding-window segmentation reconstructs full-volume output.
* LoRA injection does not accidentally target every convolution.

---

# Phase 8 — Pathology tile and WSI adapters

## Objective

Support both tile-level and slide-level models without loading an entire WSI into memory.

## Required components

```text
PathologyTileEncoder
SlideAggregator
TileSampler
EmbeddingStore
WSITokenSelector
PathologyVLMAdapter
```

## Required adapters

```text
HOptimus0Adapter
GigaPathTileAdapter
GigaPathSlideAdapter
GigaPathFlashAdapter
TITANAdapter
OptionalCONCHAdapter
GenericMILAggregator
```

## Two-stage WSI design

```text
WSI
  ↓
Tissue mask
  ↓
Tile coordinates
  ↓
Tile encoder
  ↓
Persistent embedding store
  ↓
Slide aggregator or token selector
  ↓
Classification / retrieval / VLM bridge
```

## Embedding-store format

Use Zarr, HDF5 or Parquet/Arrow-compatible structures.

Store:

```text
slide_id
tile_ids
tile_embeddings       [T, D]
tile_coordinates      [T, 2]
tile_level
tile_mpp
tile_quality
encoder_model_id
encoder_revision
preprocessing_hash
embedding_dtype
```

## GigaPath integration

The GigaPath model family separates tile encoding from slide encoding. Preserve this split in the framework. GigaPath-Flash should be the default first slide-level implementation because it is substantially lighter than running a very large tile encoder end-to-end for each training step. ([GitHub][17])

## TITAN integration

Use TITAN for:

* Slide representation extraction.
* Image-text retrieval.
* Pathology report alignment.
* VLM bridge experiments.
* Frozen slide embeddings.

TITAN was trained using whole-slide self-supervision and report alignment, making it particularly useful for slide-level multimodal experiments. ([GitHub][18])

## Tile selection policies

Implement:

```text
RandomTissueTileSelector
QualityWeightedTileSelector
DiversityTileSelector
TopKAttentionTileSelector
SpatialGridTileSelector
MultiResolutionTileSelector
TextConditionedTileSelector
```

## VLM tile-budget rules

A slide may contain tens of thousands of tiles, but the LLM must receive a bounded number of visual tokens.

Use:

```text
Raw tile count
  ↓
Tile embeddings
  ↓
Selection or aggregation
  ↓
128–1,024 selected tile embeddings
  ↓
Perceiver/Q-Former compression
  ↓
32–128 LLM visual tokens
```

The exact limits must be configurable and benchmarked. Do not make thousands of visual tokens the default on one 48GB GPU.

## Acceptance tests

* A synthetic pyramid slide can be tiled.
* Tile coordinates are stable across repeated runs.
* Cached embeddings can train a slide classifier.
* WSI VLM batches have fixed visual-token limits.
* A missing tile or corrupt region does not corrupt an entire training epoch.

---

# Phase 9 — Language models and vision-to-language bridges

## Objective

Support both native VLMs and arbitrary external visual encoders.

There should be two separate modes.

## Mode A — Native VLM mode

Use the model’s own vision tower and connector.

Examples:

* MedGemma 1.5.
* M3D-LaMed.
* FlexiCT-3D-VLM where task-compatible.

Flow:

```text
Native processor
  ↓
Native visual tower
  ↓
Native multimodal connector
  ↓
Native language model
```

## Mode B — External visual-encoder mode

Use a framework visual encoder with a configurable bridge.

Flow:

```text
2D / 3D / WSI encoder
  ↓
Spatial tokens
  ↓
Coordinate encoding
  ↓
Token projector or resampler
  ↓
Fixed visual-token budget
  ↓
Language-model embedding dimension
  ↓
LLM
```

This mode is necessary for:

* CT-FM + language model.
* Triad + language model.
* H-Optimus/GigaPath + language model.
* MedSigLIP + language model.
* Any future encoder without a native decoder.

## Bridge interface

```python
class VisionLanguageBridge(nn.Module):
    def forward(
        self,
        visual_tokens: Tensor,
        visual_mask: Tensor | None,
        coordinates: Tensor | None,
        metadata: dict[str, Any],
    ) -> ProjectedVisualTokens:
        ...
```

```python
@dataclass
class ProjectedVisualTokens:
    embeddings: Tensor       # [B, M, Dlm]
    attention_mask: Tensor   # [B, M]
    token_types: Tensor | None
    coordinates: Tensor | None
```

## Implement bridge variants

### Linear projector

```text
Dv -> Dlm
```

Use for smoke testing and small token counts.

### Two-layer MLP projector

```text
Linear(Dv, Dh)
GELU
LayerNorm
Linear(Dh, Dlm)
```

This should be the first practical default.

### Perceiver resampler

Use learned queries to compress a variable number of visual tokens into a fixed token count.

Recommended configurable values:

```yaml
num_queries: 32 | 64 | 128
num_layers: 2 | 4 | 6
num_heads: 8
```

### Q-Former-style bridge

Use for query-based visual extraction and optional text-conditioned visual token selection.

### Spatial pyramid resampler

For segmentation and small-lesion tasks:

* Preserve low-resolution global tokens.
* Select higher-resolution local tokens.
* Combine them before language projection.

## Coordinate encodings

### 2D

Include:

```text
x_normalized
y_normalized
image_index
view_position
timepoint
```

### Native 3D

Include:

```text
x_normalized
y_normalized
z_normalized
x_mm
y_mm
z_mm
voxel_spacing
series_index
```

### WSI

Include:

```text
x_slide_normalized
y_slide_normalized
microns_per_pixel
pyramid_level
slide_index
```

## Visual-token placement

Implement one common mechanism:

```text
System prompt
User text prefix
<visual_start>
Projected visual embeddings
<visual_end>
Remaining user prompt
Assistant answer
```

The framework must support model-specific variants without changing task datasets.

## LLM adapters

Implement:

```text
MedGemmaAdapter
GemmaCausalLMAdapter
GenericHFCausalLMAdapter
M3DLaMedLanguageAdapter
```

The generic adapter must verify that the chosen architecture supports either:

* `inputs_embeds`, or
* An officially supported multimodal connector API.

Do not assume all causal language models can accept arbitrary embedded tokens without architecture-specific work.

## VLM training stages

### Stage 1 — Bridge warm-up

Train:

* Bridge.
* Newly initialized visual boundary embeddings.
* Optional normalization layers.

Freeze:

* Vision encoder.
* LLM.

### Stage 2 — Language adaptation

Train:

* Bridge.
* LLM LoRA/QLoRA adapters.

Freeze:

* Vision encoder.

### Stage 3 — Vision adaptation

Train:

* Bridge.
* LLM LoRA.
* Vision LoRA in final blocks.

Only start Stage 3 after demonstrating that Stage 2 improves validation performance.

### Stage 4 — Optional multitask tuning

Train across:

* VQA.
* Structured findings.
* Classification-as-generation.
* Report generation.
* Retrieval alignment.

Use task-specific sampling weights.

## Exit criteria

* 2D external encoder to LLM produces a valid training loss.
* Native 3D encoder to LLM produces a valid training loss.
* WSI embedding set to LLM produces a valid training loss.
* Prompt tokens are masked from the loss.
* Visual tokens receive gradients through the bridge.
* Frozen visual encoders receive no gradients in Stage 1 or Stage 2.

---

# Phase 10 — LoRA, QLoRA and PEFT subsystem

## Objective

Create a reusable PEFT engine that works across visual transformers, language models and selected segmentation components.

## PEFT configuration

```yaml
peft:
  method: lora
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.05
  bias: none
  target_policy: architecture_default
  target_modules: null
  modules_to_save:
    - classifier
    - vl_bridge
  use_rslora: false
  use_dora: false
  adapter_name: task_adapter
```

PEFT supports architecture-specific LoRA configuration, target modules, rank-stabilized LoRA and other initialization strategies. ([Hugging Face][19])

## QLoRA configuration

```yaml
quantization:
  enabled: true
  method: bitsandbytes_nf4
  load_in_4bit: true
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16
```

QLoRA keeps the pretrained language-model weights quantized while training low-rank adapter parameters. ([Hugging Face][10])

## Default policy

### Language model

Start with:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Where the architecture exposes those names.

Recommended starting range:

```text
rank: 8–32
alpha: 2 × rank
dropout: 0.0–0.05
```

For a first run:

```text
rank = 16
alpha = 32
dropout = 0.05
```

### Vision transformer

Start with attention modules:

```text
qkv
query
key
value
q_proj
k_proj
v_proj
proj
out_proj
```

Then optionally MLP modules:

```text
fc1
fc2
mlp.*
```

Recommended starting range:

```text
rank: 4–16
```

### 3D Swin-style encoders

Target:

* Attention projection modules.
* Final one-third of transformer stages.
* Optional MLP layers in final stages.

Do not initially target:

* Patch embedding.
* Normalization layers.
* Convolutional stems.
* Every decoder convolution.

### Segmentation models

Use:

* Full training for a newly initialized decoder.
* LoRA for transformer encoder attention.
* Optional small adapters in skip connections.
* Full training for class-specific output layers.

### VLM bridge

Train the bridge fully. It is usually small enough that LoRA is unnecessary.

## Architecture module resolver

Implement:

```bash
medfm peft inspect --model <id>
```

Output:

```text
module_name
module_type
parameter_shape
parameter_count
suggested_lora_target
selected
reason
```

Require explicit confirmation in configuration for unknown model families.

## Trainable-parameter audit

Before every run, print:

```text
Total parameters
Frozen parameters
Trainable adapter parameters
Trainable bridge parameters
Trainable task-head parameters
Trainable decoder parameters
Trainable percentage
```

Fail training if:

* `trainable_parameters == 0`.
* Full LLM weights are unexpectedly trainable in QLoRA mode.
* Quantized parameters are placed in the optimizer.
* A supposedly frozen visual encoder has gradients.

## Multiple adapters

Support namespaced adapters:

```text
vision_adapter
language_adapter
classification_adapter
segmentation_adapter
site_adapter
modality_adapter
```

Checkpoint structure:

```text
checkpoint/
├── manifest.json
├── base_models.json
├── preprocessing.yaml
├── task.yaml
├── bridge/
│   └── model.safetensors
├── adapters/
│   ├── vision_adapter/
│   └── language_adapter/
├── heads/
│   └── task_head.safetensors
└── metrics.json
```

## Merge policy

* Do not merge during training.
* Save adapter-only checkpoints.
* Allow optional merged export for inference.
* Preserve an unmerged adapter artifact.
* Record the exact base-model revision.
* Validate output equivalence before and after merging.

TRL and PEFT support keeping adapters separate or merging them for inference. ([Hugging Face][20])

## Exit criteria

* LoRA can be attached to one 2D ViT.
* LoRA can be attached to one 3D transformer.
* QLoRA can be attached to the selected 4B language model.
* Separate visual and language adapters save and reload.
* An adapter merge equivalence test passes within tolerance.

---

# Phase 11 — Task heads, decoders and loss functions

## Objective

Implement task-specific modules independently of individual foundation models.

## Classification heads

Implement:

```text
LinearClassificationHead
MLPClassificationHead
AttentionPoolingClassificationHead
MultiLabelClassificationHead
OrdinalClassificationHead
MILClassificationHead
```

Pooling options:

```text
CLS
Mean valid token pooling
Attention pooling
Generalized mean pooling
Top-k pooling
MIL attention pooling
```

## Classification losses

Support:

* Binary cross-entropy with logits.
* Multiclass cross-entropy.
* Class-weighted variants.
* Focal loss.
* Label smoothing.
* Asymmetric multilabel loss.
* Ordinal cumulative-link loss.

The baseline must always include ordinary BCE or cross-entropy before more elaborate losses.

## Segmentation decoders

Implement common interfaces for:

```text
UNetDecoder2D
UNetDecoder3D
FPNDecoder2D
FPNDecoder3D
TransformerMaskDecoder
PromptableMaskDecoder
LanguageConditionedMaskDecoder
NativeModelDecoderWrapper
```

## Segmentation losses

Default:

```text
Dice loss + cross-entropy
```

For binary segmentation:

```text
Dice loss + BCEWithLogits
```

Optional:

* Focal loss.
* Tversky loss.
* Boundary loss.
* Deep-supervision loss.
* Class-volume weighting.

## Language-conditioned segmentation

Flow:

```text
Text prompt
  ↓
Text encoder
  ↓
Text embeddings
  ↓
Cross-attention with visual feature maps
  ↓
Mask query
  ↓
2D or 3D segmentation decoder
```

Do not generate pixel masks as raw text tokens for the primary segmentation path. The LLM may produce labels, prompts or region descriptions, while the spatial decoder produces masks.

## Retrieval head

Support:

* Image projection.
* Text projection.
* L2 normalization.
* Learnable logit scale.
* Symmetric contrastive loss.
* Distributed-negative support for future multi-GPU use.
* Same-patient negative filtering where needed.

## Localization head

Support:

* 2D boxes.
* 3D boxes.
* Normalized coordinates.
* Physical-coordinate conversion.
* L1 loss.
* GIoU/IoU-style loss.
* Structured text representation for VLM supervision.

## Structured-generation head

Define JSON schemas for generated findings:

```json
{
  "findings": [
    {
      "label": "string",
      "status": "present|absent|uncertain",
      "anatomy": "string",
      "laterality": "left|right|bilateral|midline|none",
      "severity": "string|null",
      "location": "string|null"
    }
  ],
  "impression": "string"
}
```

Validate generated JSON before scoring.

## Multitask loss

```python
total_loss = (
    w_cls * classification_loss
    + w_seg * segmentation_loss
    + w_lm * language_loss
    + w_align * contrastive_loss
    + w_box * localization_loss
)
```

Support:

* Fixed weights.
* Scheduled weights.
* Uncertainty weighting.
* GradNorm-style weighting later.

Start with fixed weights for reproducibility.

## Exit criteria

* Every task head accepts `EncoderOutput`.
* A head never depends on a specific encoder class.
* 2D and 3D segmentation use the same task interface.
* Classification, segmentation and language losses can coexist in one step.
* Invalid structured VLM output is reported rather than silently scored.

---

# Phase 12 — Unified training engine and 48GB memory planner

## Objective

Create one trainer with task-specific step functions and explicit memory controls.

## Trainer architecture

```text
RunConfig
  ↓
Registry
  ↓
Dataset builder
  ↓
Model builder
  ↓
PEFT injector
  ↓
Optimizer builder
  ↓
Task module
  ↓
Trainer
  ↓
Evaluator
  ↓
Checkpoint manager
```

## Main trainer interfaces

```python
class Trainer:
    def train(self) -> TrainingResult: ...
    def validate(self) -> EvaluationResult: ...
    def save_checkpoint(self) -> Path: ...
    def resume(self, checkpoint: Path) -> None: ...
```

```python
class TrainingStep:
    def forward_and_loss(
        self,
        model: nn.Module,
        batch: MedicalBatch,
    ) -> LossOutput:
        ...
```

## Use Accelerate for

* Mixed precision.
* Gradient accumulation.
* Device placement.
* Gradient clipping.
* Logging hooks.
* Checkpoint coordination.
* Future multi-GPU compatibility.

Accelerate provides explicit gradient-accumulation support and allows a training loop to remain compatible with different execution backends. ([Hugging Face][21])

## Precision policy

Preferred:

```text
BF16 for forward/backward where hardware supports it
FP32 for selected numerically sensitive losses and metrics
NF4 storage for QLoRA LLM weights
FP32 optimizer state for small adapters when practical
```

## Activation checkpointing

Expose:

```yaml
memory:
  gradient_checkpointing:
    language_model: true
    vision_encoder: false
    bridge: false
    segmentation_decoder: false
```

Activation checkpointing reduces saved activation memory by recomputing selected activations during backward. ([PyTorch][22])

## Attention policy

Priority:

1. PyTorch scaled-dot-product attention.
2. FlashAttention where architecture-compatible.
3. Eager attention fallback.

PyTorch SDPA automatically chooses compatible optimized implementations, while FlashAttention supplies explicit memory-efficient kernels for supported architectures. ([PyTorch Documentation][23])

## Memory-planner configuration

```yaml
memory:
  max_gpu_memory_gb: 46
  reserve_gpu_memory_gb: 2

  micro_batch_size: 1
  gradient_accumulation_steps: 16

  activation_checkpointing: true
  use_cache_during_training: false

  visual_token_budget: 64
  max_text_tokens: 1024

  empty_cache_on_validation: false
  log_peak_memory: true
```

Do not try to consume all 48GB. Reserve approximately 2GB for runtime variation, kernels and fragmentation.

## OOM recovery policy

The first OOM should produce a diagnostic, not immediately retry indefinitely.

Recommended adjustment order:

1. Reduce microbatch to 1.
2. Enable activation checkpointing.
3. Disable language-model KV cache.
4. Reduce maximum text length.
5. Reduce visual-token budget.
6. Reduce number of 2D images or slices.
7. Reduce 3D patch dimensions.
8. Freeze visual encoder.
9. Cache visual embeddings.
10. Enable optimizer-state reduction.
11. Use CPU offload only as an explicit last resort.

The framework may suggest a new configuration, but it should not silently alter an experiment and continue under a different regime.

## Optimizer groups

Separate learning rates:

```yaml
optimizer_groups:
  bridge:
    lr: 1.0e-4
  task_head:
    lr: 1.0e-4
  segmentation_decoder:
    lr: 1.0e-4
  vision_lora:
    lr: 2.0e-5
  language_lora:
    lr: 1.0e-5
```

These are starting points, not universal optima.

## Freeze schedules

Support:

```yaml
freeze_schedule:
  - until_step: 2000
    train:
      - bridge
      - task_head
  - until_step: 10000
    train:
      - bridge
      - task_head
      - language_lora
  - until_step: null
    train:
      - bridge
      - task_head
      - language_lora
      - vision_lora
```

## Checkpoint contents

Save:

* Adapter weights.
* Bridge.
* Task head.
* Segmentation decoder.
* Optimizer.
* Scheduler.
* Gradient scaler if used.
* Epoch and step.
* RNG state.
* Dataloader sampler state where possible.
* Configuration.
* Dataset hash.
* Base-model references.
* Evaluation metrics.
* Best-checkpoint criterion.

## Required trainer tests

### One-batch overfit

Every recipe must be capable of overfitting one tiny batch.

### Resume equivalence

* Train N steps.
* Save.
* Reload.
* Train one more step.
* Compare with uninterrupted training.

### Gradient audit

Check that only expected modules have gradients.

### Memory audit

Record:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

## Exit criteria

* One classification run.
* One segmentation run.
* One 2D VLM run.
* One 3D VLM run.
* Each completes at least one optimizer step on the target GPU.
* Checkpoint resume works.
* Peak VRAM is logged.

---

# Phase 13 — 2D training recipes

## Objective

Provide production-quality recipe templates for 2D classification, segmentation and VLM training.

---

## Recipe 13A — 2D classification

### Supported encoders

* MedSigLIP.
* RAD-DINO.
* H-Optimus-0 for pathology tiles.
* MedGemma vision tower where extractable.
* FlexiCT-2D for CT slices.

### Stage A — Frozen linear probe

Train:

* Pooling layer.
* Classification head.

Freeze:

* Entire visual encoder.

Purpose:

* Establish data and label correctness.
* Establish a minimum baseline.
* Detect whether foundation-model features transfer.

### Stage B — Partial LoRA

Train:

* Head.
* LoRA in final 25–33% of visual transformer blocks.

### Stage C — Broader LoRA

Only after Stage B:

* Add MLP LoRA.
* Increase rank if underfitting.
* Optionally unfreeze final normalization layer.

### Example config

```yaml
task:
  type: multilabel_classification
  num_classes: 14
  loss: bce_with_logits

model:
  visual_encoder: medsiglip_448
  pooling: cls

peft:
  method: lora
  rank: 8
  alpha: 16
  target_policy: final_quarter_attention

trainer:
  precision: bf16
  micro_batch_size: 8
  gradient_accumulation_steps: 4
```

Actual batch size depends on input resolution, model implementation and enabled hidden-state outputs.

---

## Recipe 13B — 2D segmentation

### Supported encoders

* RAD-DINO.
* MedSigLIP patch tokens.
* H-Optimus patch tokens for tile-level pathology segmentation.
* MedSAM2.
* Generic 2D transformer adapters.

### Training order

1. Freeze encoder.
2. Train decoder.
3. Add vision LoRA in late blocks.
4. Optionally fine-tune prompt encoder or mask decoder for promptable models.
5. Validate at original image resolution.

### Required outputs

* Per-class Dice.
* Surface metrics where applicable.
* Lesion-wise sensitivity.
* False positives per image.
* Original-space masks.

---

## Recipe 13C — Native 2D VLM

### Primary model

* MedGemma 1.5 4B.

### Stage 1

* QLoRA language model.
* Train native multimodal connector where permitted.
* Keep visual tower frozen.
* Train on VQA or structured findings before free-form report generation.

### Stage 2

* Mix:

  * VQA.
  * Classification-as-generation.
  * Structured findings.
  * Short impression generation.
  * Full report generation.

### Stage 3

* Optional vision LoRA.
* Small learning rate.
* Strong validation against frozen-vision Stage 2.

### Recommended starting settings

```yaml
quantization:
  load_in_4bit: true
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16

peft:
  rank: 16
  alpha: 32
  dropout: 0.05

trainer:
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  max_text_tokens: 1024
  gradient_checkpointing: true
  use_cache_during_training: false
```

---

## Recipe 13D — External-encoder 2D VLM

Example:

```text
MedSigLIP / RAD-DINO / H-Optimus
  ↓
Patch tokens
  ↓
Perceiver resampler
  ↓
64 visual tokens
  ↓
Gemma/MedGemma-compatible language adapter
```

Training stages:

1. Frozen vision + frozen LLM + bridge training.
2. Frozen vision + LLM QLoRA + bridge.
3. Vision LoRA + LLM QLoRA + bridge.

## Exit criteria

* Frozen classification baseline exists.
* LoRA classification baseline exists.
* Frozen-encoder segmentation baseline exists.
* Native 2D VLM baseline exists.
* External-encoder 2D VLM baseline exists.

---

# Phase 14 — 3D CT/MRI training recipes

## Objective

Support both true volumetric reasoning and slice-sequence reasoning.

---

## Recipe 14A — Native 3D classification

### Encoders

* CT-FM.
* FlexiCT-3D.
* Merlin.
* Triad.
* M3D-CLIP.

### Input strategies

Support:

```text
Full resampled volume
Fixed 3D crop
Multiple sampled crops
Low-resolution global volume
Global + high-resolution local crops
```

### Recommended starting approach

```text
Low-resolution whole-volume crop
or
96³–128³ task-specific crop
```

The exact crop must be selected by the dataset fingerprint and encoder constraints.

### Training stages

1. Frozen encoder + pooled classification head.
2. Frozen encoder + attention pooling.
3. Final-stage vision LoRA.
4. Multi-crop aggregation if needed.

### Multi-crop aggregation

```text
Volume
  ↓
K crops
  ↓
K embeddings
  ↓
Attention pooling
  ↓
Prediction
```

---

## Recipe 14B — Native 3D segmentation

### Encoders

* CT-FM.
* Triad.
* NV-Segment-CTMR.
* Generic MONAI 3D encoders.
* FlexiCT-3D where spatial feature resolution is sufficient.

### Training stages

1. Establish an nnU-Net or MONAI baseline.
2. Freeze foundation encoder and train decoder.
3. Add LoRA to later encoder blocks.
4. Add deep supervision if beneficial.
5. Validate with sliding-window inference.
6. Invert all transforms into original space.

MONAI’s sliding-window inferers are specifically designed to process large 2D or 3D inputs in smaller regions and combine the outputs. ([MONAI][24])

### Required sampling behavior

For small lesions:

* At least one configurable fraction of patches must intersect the target.
* Log positive-patch rate.
* Log lesion-size distribution.
* Evaluate lesion-wise recall, not only global Dice.

---

## Recipe 14C — Native 3D VLM

### Architecture

```text
CT-FM / FlexiCT-3D / Merlin / Triad
  ↓
3D spatial tokens
  ↓
3D coordinate embeddings
  ↓
Perceiver resampler
  ↓
32–128 visual tokens
  ↓
Language-model embedding space
  ↓
QLoRA language model
```

### Stage 1 — 3D-to-text alignment

Train:

* 3D bridge.
* Optional contrastive projections.

Freeze:

* 3D encoder.
* LLM.

Tasks:

* Report/volume matching.
* Image-text retrieval.
* Finding-tag prediction.
* Short structured descriptions.

### Stage 2 — Instruction tuning

Train:

* Bridge.
* LLM QLoRA.

Freeze:

* 3D encoder.

Tasks:

* VQA.
* Condition classification as structured text.
* Anatomy identification.
* Structured findings.
* Impression generation.

### Stage 3 — Vision adaptation

Train:

* Bridge.
* LLM QLoRA.
* LoRA in final 3D encoder stages.

### Stage 4 — Grounding

Add:

* Bounding-box supervision.
* Region-token selection.
* Language-conditioned segmentation.
* Anatomical coordinate output.

### 48GB starting profile

```yaml
model:
  visual_encoder: ct_fm
  language_model: medical_4b
  bridge:
    type: perceiver
    num_visual_tokens: 64
    layers: 4

data:
  volume_shape: [96, 128, 128]
  samples_per_volume: 1

trainer:
  precision: bf16
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  max_text_tokens: 512
  gradient_checkpointing:
    language_model: true
    visual_encoder: false
```

If the 3D encoder remains too expensive:

* Cache its spatial tokens.
* Train the bridge and LLM adapters from cached tokens.
* Later run a short joint vision-LoRA phase with smaller crops.

---

## Recipe 14D — Slice-sequence VLM

This is distinct from native 3D VLM training.

### Flow

```text
CT/MRI volume
  ↓
Slice selection
  ↓
2D image encoder or native multi-image processor
  ↓
Per-slice embeddings
  ↓
Slice-position embeddings
  ↓
Sequence compression
  ↓
LLM
```

### Slice selectors

Implement:

```text
UniformSliceSelector
AnatomyAwareSliceSelector
ReportConditionedSliceSelector
HighEntropySliceSelector
LesionAwareSliceSelector
MultiWindowSliceSelector
```

### Required positional information

* Slice index.
* Normalized z position.
* Physical z coordinate.
* Series order.
* Window/channel type.
* Sequence identifier for MRI.

### Recommended stage order

1. Uniform slice sampling.
2. Fixed number of slices.
3. Frozen visual tower.
4. Projector training.
5. LLM QLoRA.
6. Learnable slice selection later.

MedGemma 1.5’s high-dimensional imaging pathway is based on long-context collections of 2D images, so this recipe is the natural way to reproduce or extend that class of functionality. ([arXiv][25])

---

## Recipe 14E — Language-conditioned 3D segmentation

```text
3D image
  ↓
3D encoder feature pyramid
                    Text prompt
                         ↓
                  Text embeddings
                         ↓
Feature/text cross-attention
  ↓
3D segmentation decoder
  ↓
Mask
```

Examples:

```text
“Segment the left kidney.”
“Segment all hepatic lesions.”
“Segment the enhancing tumor component.”
```

The mask remains a spatial decoder output, while language provides the class or region query.

## Exit criteria

* 3D classification works from a native volume.
* 3D segmentation works with sliding-window inference.
* Native 3D tokens can condition an LLM.
* Slice-sequence VLM can condition an LLM.
* Native 3D and slice-sequence experiments use separate configuration names and metrics.

---

# Phase 15 — Pathology task recipes

## Objective

Support tile, slide and multimodal pathology training under one GPU.

---

## Recipe 15A — Tile classification

### Encoders

* H-Optimus-0.
* CONCH.
* GigaPath tile encoder.
* MedSigLIP where appropriate.

### Stages

1. Frozen tile encoder + linear head.
2. Frozen tile encoder + MLP.
3. Vision LoRA in final blocks.
4. Optional contrastive text alignment.

---

## Recipe 15B — WSI classification

### Default flow

```text
Precomputed tile embeddings
  ↓
GigaPath-Flash or MIL aggregator
  ↓
Slide embedding
  ↓
Classification head
```

### Aggregators

* Mean pooling baseline.
* Attention MIL.
* Gated attention MIL.
* Transformer slide encoder.
* GigaPath-Flash.
* TITAN embeddings.

### Rules

* Always retain tile coordinates.
* Always compare against mean pooling.
* Use patient-level splits.
* Log number of tiles per slide.
* Bound tiles per training batch.
* Use random or importance sampling during training.
* Use deterministic tile policy during evaluation.

---

## Recipe 15C — WSI VLM

### Architecture

```text
WSI
  ↓
Tile embeddings
  ↓
Slide aggregator / tile selector
  ↓
Coordinate-aware resampler
  ↓
32–128 visual tokens
  ↓
QLoRA LLM
```

### Training stages

1. Frozen tile and slide encoders.
2. Train WSI bridge.
3. Add LLM QLoRA.
4. Fine-tune slide aggregator.
5. Add tile-encoder LoRA only for narrowly scoped experiments.

### Tasks

* Organ/site identification.
* Histologic subtype classification.
* Grade prediction.
* Biomarker prediction.
* Report-section generation.
* Pathology VQA.
* Image-text retrieval.
* Evidence-tile selection.

### Evidence supervision

Where available, return:

```json
{
  "answer": "...",
  "evidence_tiles": [
    {"x": 0.42, "y": 0.61, "score": 0.91}
  ]
}
```

This provides a bridge between slide-level language output and spatial evidence.

---

## Recipe 15D — Pathology segmentation

For ROI-annotated pathology:

* Use tile-level image and mask pairs.
* Train a 2D segmentation decoder.
* Stitch tile predictions.
* Blend overlaps.
* Map masks to slide coordinates.
* Evaluate both tile and slide levels.

Do not train gigapixel segmentation end-to-end.

## Exit criteria

* Cached H-Optimus or GigaPath embeddings train a slide classifier.
* GigaPath-Flash or a MIL aggregator produces slide embeddings.
* WSI visual tokens condition an LLM.
* Evidence tiles map back to original WSI coordinates.
* Pathology segmentation can reconstruct a slide-level mask.

---

# Phase 16 — Evaluation, clinical validation and regression testing

## Objective

Build an evaluation layer that measures more than a single headline metric.

## Classification metrics

Report:

* AUROC.
* Average precision/AUPRC.
* Sensitivity.
* Specificity.
* Precision.
* Recall.
* F1.
* Balanced accuracy.
* Confusion matrix.
* Calibration error.
* Brier score.
* Sensitivity at fixed specificity.
* Specificity at fixed sensitivity.

TorchMetrics provides standard implementations for AUROC, average precision and calibration-related classification metrics. ([Lightning AI][26])

### Reporting rules

* Macro and micro averages for multilabel tasks.
* Per-class confidence intervals.
* Patient-level bootstrap.
* No slice-level confidence intervals when the clinical unit is the patient or study.
* Thresholds selected on validation only.

## Segmentation metrics

Report:

* Dice.
* IoU.
* Surface Dice.
* Hausdorff distance or HD95.
* Average symmetric surface distance.
* Lesion-wise sensitivity.
* False-positive lesions per scan.
* Volume error.
* Per-class and macro summaries.

Evaluate in original physical space where practical.

## Retrieval metrics

* Recall@1.
* Recall@5.
* Recall@10.
* Median rank.
* Mean rank.
* mAP.
* Separate image-to-text and text-to-image results.

## VQA and structured output

* Exact match.
* Token-level F1.
* Schema validity.
* Finding-level precision/recall.
* Negation correctness.
* Laterality correctness.
* Severity correctness.
* Anatomy correctness.

## Report-generation evaluation

Do not rely only on BLEU or ROUGE.

Use:

* RadGraph-based entity/relation scoring.
* Clinical finding extraction.
* Contradiction detection.
* Omission analysis.
* Hallucinated finding rate.
* Human expert review on a defined subset.

RadGraph extracts clinical entities and relations from radiology reports and is more clinically informative than purely lexical overlap metrics. ([GitHub][27])

## 3D VLM evaluation

Evaluate separately:

* Whole-volume classification.
* Slice-specific findings.
* Small-lesion sensitivity.
* Anatomy localization.
* 3D box localization.
* Report generation.
* Spatial consistency across adjacent slices.
* Performance against the same model using slice-sequence inputs.

## Pathology evaluation

Report:

* Tile-level metrics.
* Slide-level metrics.
* Patient-level metrics.
* Organ-specific results.
* Scanner/site-specific results.
* Attention/evidence localization.
* Performance versus number of sampled tiles.
* Performance versus magnification.

## Generalization tests

At minimum:

* Internal random holdout.
* Patient-disjoint holdout.
* External-site holdout where possible.
* Temporal holdout where possible.
* Scanner/vendor subgroup.
* Acquisition-protocol subgroup.
* Rare-class subgroup.
* Missing-sequence MRI subgroup.
* Low-quality WSI subgroup.

## Baselines

Every task should include:

* Random or majority baseline.
* Frozen foundation-model linear probe.
* LoRA fine-tune.
* Conventional task-specific baseline.
* Full decoder with frozen encoder for segmentation.
* nnU-Net or comparable baseline for 3D segmentation where applicable.

## Ablations

For VLMs:

* No visual input.
* Shuffled visual input.
* Frozen bridge.
* Linear versus Perceiver bridge.
* 32 versus 64 versus 128 visual tokens.
* Frozen versus LoRA visual encoder.
* Native 3D versus slice sequence.
* With versus without coordinate embeddings.

A VLM that performs similarly with shuffled visual inputs has not learned adequate visual grounding.

## Human review template

For report or VQA output, capture:

```text
Correct
Minor error
Major error
Potentially harmful error
Unsupported finding
Omitted critical finding
Incorrect negation
Incorrect laterality
Incorrect severity
Incorrect anatomy
Poor uncertainty expression
```

## Exit criteria

* Every recipe has a metric suite.
* Every metric is computed at the appropriate clinical unit.
* VLM ablations confirm visual dependence.
* Every release candidate has subgroup results and error examples.
* No model is labeled clinically validated without an explicit validation study.

---

# Phase 17 — Inference, export and serving

## Objective

Make trained adapters reproducible and usable without coupling them to the training repository internals.

## Export format

Use a framework bundle inspired by MONAI Bundles:

```text
exported_model/
├── bundle.json
├── model_card.md
├── license_summary.md
├── base_models.json
├── preprocessing.yaml
├── postprocessing.yaml
├── task_schema.json
├── inference_config.yaml
├── adapters/
├── bridge/
├── heads/
├── calibration/
├── examples/
└── checksums.json
```

MONAI Bundles are specifically designed to package weights, metadata, configurations and reproducibility information together. ([MONAI][1])

## Inference commands

```bash
medfm infer classification --config ...
medfm infer segmentation --config ...
medfm infer vlm --config ...
medfm infer retrieval --config ...
medfm infer wsi --config ...
```

## 3D segmentation inference

* Sliding-window execution.
* Configurable overlap.
* Gaussian blending.
* Test-time augmentation optional.
* Restore original orientation.
* Restore original spacing.
* Export NIfTI.
* Optional DICOM SEG export.

## DICOM output

Use highdicom for interoperable derived objects where appropriate:

* DICOM SEG.
* DICOM SR.
* Parametric maps where applicable.

highdicom is designed to simplify the creation and interpretation of image-derived DICOM objects for radiology and pathology applications. ([Highdicom][28])

## VLM inference

Support:

* Deterministic greedy generation.
* Beam search where justified.
* Temperature-controlled sampling for research.
* JSON-constrained decoding.
* Maximum output limits.
* Stop tokens.
* Prompt-version recording.
* Optional uncertainty response.

For evaluation and clinical-style output, default to deterministic decoding.

## Adapter serving

Allow:

```text
One base LLM
+ multiple task LoRA adapters
+ multiple visual bridges
```

Example:

```text
MedGemma base
├── chest_xray_report_adapter
├── ct_vqa_adapter
├── mri_classification_generation_adapter
└── pathology_report_adapter
```

Do not load all adapters simultaneously unless required.

## Service architecture

```text
API
  ↓
Request validator
  ↓
Modality router
  ↓
Preprocessor
  ↓
Model/adapter loader
  ↓
Inference
  ↓
Postprocessor
  ↓
Structured result
  ↓
Audit log
```

## Required audit fields

* Model ID and revision.
* Adapter ID and revision.
* Preprocessing hash.
* Prompt version.
* Input hash.
* Timestamp.
* Output schema version.
* Runtime.
* Peak VRAM.
* Error status.

Do not log unredacted reports or images by default.

## Exit criteria

* Adapter-only export loads in a clean environment.
* 3D masks return to original coordinates.
* VLM output passes schema validation.
* DICOM-derived output can be reopened.
* Inference memory remains below the configured cap.

---

# Phase 18 — CI, hardening and release

## Objective

Prevent silent regressions across a highly heterogeneous model stack.

## Test levels

### Level 1 — CPU unit tests

* Schemas.
* Registries.
* Configuration validation.
* Loss functions.
* Metric accumulation.
* Prompt formatting.
* Cache keys.
* Coordinate transforms.

### Level 2 — Synthetic GPU tests

* One 2D encoder.
* One 3D encoder.
* One pathology encoder.
* One LLM QLoRA model, potentially a tiny test model.
* One segmentation decoder.
* One VLM bridge.

### Level 3 — Real-checkpoint smoke tests

Run manually or in a protected GPU environment:

* MedSigLIP.
* CT-FM or FlexiCT.
* Triad or NV-Segment.
* H-Optimus or GigaPath.
* MedGemma.

### Level 4 — Golden regression tests

Maintain small, de-identified or synthetic golden cases with expected:

* Tensor shapes.
* Preprocessing statistics.
* Classification logits within tolerance.
* Segmentation mask checksum/tolerance.
* Generated structured fields.
* Peak-memory envelope.

## Security tests

* Path traversal in manifests.
* Arbitrary code execution from untrusted model repositories.
* Unsafe `trust_remote_code`.
* Malicious checkpoint files.
* PHI in logs.
* PHI in exception messages.
* Prompt injection inside reports.
* Unauthorized model download.
* License-policy bypass.

Default:

```text
trust_remote_code = false
```

Enable it only for reviewed and pinned repositories.

## Release gates

A release requires:

* Passing tests.
* Model registry complete.
* License summary complete.
* Model card complete.
* Training configuration archived.
* Validation report archived.
* Known limitations listed.
* Data provenance documented.
* No unsupported clinical claims.

---

# Recommended repository layout

```text
medfm/
├── cli/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── models.py
│   ├── data.py
│   └── export.py
│
├── core/
│   ├── enums.py
│   ├── schemas.py
│   ├── protocols.py
│   ├── capabilities.py
│   ├── exceptions.py
│   └── registry.py
│
├── data/
│   ├── manifests/
│   ├── readers/
│   │   ├── dicom.py
│   │   ├── nifti.py
│   │   ├── image2d.py
│   │   ├── wsi.py
│   │   └── embeddings.py
│   ├── transforms/
│   │   ├── common.py
│   │   ├── radiology2d.py
│   │   ├── ct3d.py
│   │   ├── mri3d.py
│   │   └── pathology.py
│   ├── samplers/
│   ├── collators/
│   ├── caching/
│   └── fingerprint.py
│
├── models/
│   ├── visual/
│   │   ├── base.py
│   │   ├── medsiglip.py
│   │   ├── rad_dino.py
│   │   ├── ct_fm.py
│   │   ├── flexict.py
│   │   ├── merlin.py
│   │   ├── triad.py
│   │   ├── nv_segment.py
│   │   ├── medsam2.py
│   │   ├── h_optimus.py
│   │   ├── gigapath.py
│   │   └── titan.py
│   │
│   ├── language/
│   │   ├── base.py
│   │   ├── medgemma.py
│   │   ├── gemma.py
│   │   └── generic_hf.py
│   │
│   ├── bridges/
│   │   ├── linear.py
│   │   ├── mlp.py
│   │   ├── perceiver.py
│   │   ├── qformer.py
│   │   └── spatial_pyramid.py
│   │
│   ├── heads/
│   │   ├── classification.py
│   │   ├── retrieval.py
│   │   ├── localization.py
│   │   └── structured_generation.py
│   │
│   └── decoders/
│       ├── segmentation2d.py
│       ├── segmentation3d.py
│       ├── promptable.py
│       └── language_conditioned.py
│
├── peft/
│   ├── config.py
│   ├── injector.py
│   ├── resolver.py
│   ├── quantization.py
│   ├── audit.py
│   ├── merge.py
│   └── checkpoint.py
│
├── tasks/
│   ├── classification.py
│   ├── segmentation.py
│   ├── retrieval.py
│   ├── vlm_sft.py
│   ├── contrastive.py
│   └── multitask.py
│
├── training/
│   ├── trainer.py
│   ├── steps.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── memory.py
│   ├── checkpoint.py
│   └── tracking.py
│
├── evaluation/
│   ├── classification.py
│   ├── segmentation.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── grounding.py
│   ├── calibration.py
│   └── bootstrap.py
│
└── inference/
    ├── pipeline.py
    ├── sliding_window.py
    ├── generation.py
    ├── export_nifti.py
    ├── export_dicom.py
    └── server.py
```

---

# Unified experiment configuration

A representative 3D VLM configuration should look like:

```yaml
run:
  name: ct_fm_medical4b_vqa
  seed: 42
  output_dir: artifacts/runs/ct_fm_medical4b_vqa

data:
  manifest: datasets/ct_vqa/train.parquet
  modality: CT_3D

  preprocessing:
    orientation: RAS
    spacing_mm: [2.0, 2.0, 2.0]
    hu_range: [-1000, 1000]
    output_shape: [96, 128, 128]

  text:
    max_input_tokens: 512
    max_output_tokens: 512
    template: medical_vqa_v1

model:
  visual:
    id: ct_fm
    revision: pinned_revision
    frozen: true
    output_layer: final
    output_spatial_tokens: true

  bridge:
    type: perceiver
    num_queries: 64
    hidden_dim: 1024
    num_layers: 4
    coordinate_encoding: physical_3d

  language:
    id: medical_4b
    revision: pinned_revision
    use_cache: false

peft:
  vision:
    enabled: false

  language:
    enabled: true
    method: lora
    rank: 16
    alpha: 32
    dropout: 0.05
    target_policy: decoder_attention_and_mlp

quantization:
  language:
    enabled: true
    method: nf4
    double_quant: true
    compute_dtype: bfloat16

task:
  type: visual_question_answering
  loss: causal_language_modeling
  mask_prompt_tokens: true

trainer:
  precision: bf16
  epochs: 3
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing:
    language: true
    visual: false
  gradient_clip_norm: 1.0

optimizer:
  type: adamw
  groups:
    bridge:
      lr: 1.0e-4
    language_lora:
      lr: 1.0e-5
  weight_decay: 0.01

evaluation:
  metrics:
    - exact_match
    - token_f1
    - schema_validity
    - finding_f1
  generation:
    deterministic: true
    max_new_tokens: 512

checkpoint:
  save_adapter_only: true
  save_every_steps: 500
  keep_last: 3
  monitor: validation/finding_f1
  mode: max
```

---

# Single-48GB starting configurations

These are conservative starting points, not guaranteed capacity figures. Actual memory depends on architecture, crop dimensions, sequence lengths, hidden-state extraction and CUDA kernels.

| Workload                      | Starting strategy                                             |
| ----------------------------- | ------------------------------------------------------------- |
| MedSigLIP classification      | BF16, frozen encoder first, batch 4–16, then visual LoRA      |
| RAD-DINO classification       | BF16, frozen or visual LoRA, batch 8–32 depending resolution  |
| H-Optimus tile classification | Frozen BF16 or cached embeddings; batch 4–16                  |
| 2D segmentation               | Frozen encoder, full decoder, 512² or model-native resolution |
| 3D classification             | Batch 1, 96³–128³ crop, frozen encoder first                  |
| 3D segmentation               | Batch 1, patch-based, sliding-window validation               |
| MedGemma 4B VLM               | NF4 QLoRA, batch 1–2, gradient accumulation, checkpointing    |
| External 2D VLM               | Frozen vision, 32–128 visual tokens, LLM QLoRA                |
| External 3D VLM               | Batch 1, 32–64 visual tokens initially, 512-token text limit  |
| M3D-LaMed-style 7B            | NF4 QLoRA, batch 1, reduced volume and text length            |
| WSI classification            | Cached embeddings, 512–4,096 sampled tiles                    |
| WSI VLM                       | Cached tile embeddings, 32–128 compressed visual tokens       |

---

# Recommended implementation milestones

## Milestone 0 — Skeleton

Complete Phases 0–2.

Deliverable:

* Repository.
* Schemas.
* Interfaces.
* Agent workflow.
* No real model integration yet.

## Milestone 1 — First 2D classification

Complete:

* MedSigLIP adapter.
* Frozen classification head.
* LoRA classification.
* Evaluation.

This validates the registry, dataset, trainer, PEFT and checkpoint systems with the simplest workload.

## Milestone 2 — First 3D classification

Complete:

* CT-FM or FlexiCT-3D adapter.
* CT preprocessing.
* Patch sampler.
* Frozen head.
* Vision LoRA.

## Milestone 3 — First 3D segmentation

Complete:

* CT-FM/Triad feature extraction.
* 3D decoder.
* Sliding-window inference.
* Original-space reconstruction.
* nnU-Net baseline.

## Milestone 4 — Native 2D VLM

Complete:

* MedGemma QLoRA.
* VQA.
* Structured findings.
* Report-generation evaluation.

## Milestone 5 — External 2D encoder VLM

Complete:

* MedSigLIP or RAD-DINO.
* Perceiver bridge.
* LLM QLoRA.
* Visual-dependence ablations.

## Milestone 6 — Native 3D VLM

Complete:

* CT-FM/FlexiCT/Merlin encoder.
* 3D coordinate embeddings.
* Perceiver bridge.
* LLM QLoRA.
* VQA and structured findings.

## Milestone 7 — Slice-sequence VLM

Complete:

* Volume slice selector.
* Positional metadata.
* Multi-image collator.
* MedGemma-style training.

## Milestone 8 — Pathology

Complete:

* H-Optimus tile embeddings.
* GigaPath-Flash or MIL slide model.
* WSI classification.
* WSI VLM bridge.

## Milestone 9 — Multitask system

Complete:

* Classification.
* Segmentation.
* Retrieval.
* VLM.
* Shared encoder with separate adapters.
* Adapter switching.

---

# First implementation priority

The first end-to-end development sequence should be:

1. Core schemas and registry.
2. DICOM/NIfTI and ordinary-image ingestion.
3. MedSigLIP 2D classification.
4. CT-FM 3D classification.
5. CT-FM or Triad 3D segmentation.
6. MedGemma 1.5 QLoRA VLM.
7. MedSigLIP-to-LLM external bridge.
8. CT-FM-to-LLM native 3D bridge.
9. H-Optimus embedding cache.
10. GigaPath-Flash WSI classification.
11. WSI-to-LLM bridge.
12. MedSAM2 and NV-Segment task wrappers.
13. Multitask scheduling.
14. Export and DICOM interoperability.

This order verifies each subsystem independently before combining the most memory-intensive components.

---

# Primary reference links

## Core frameworks

* [MONAI](https://project-monai.github.io/)
* [MONAI Bundles](https://monai.readthedocs.io/en/latest/bundle_intro.html)
* [Hugging Face PEFT](https://huggingface.co/docs/peft/index)
* [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)
* [Hugging Face TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
* [Transformers bitsandbytes quantization](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
* [PyTorch activation checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/)
* [FlashAttention](https://github.com/Dao-AILab/flash-attention)

## Medical VLM and image-text models

* [MedGemma documentation](https://developers.google.com/health-ai-developer-foundations/medgemma)
* [MedGemma 1.5 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
* [MedGemma GitHub](https://github.com/google-health/medgemma)
* [MedSigLIP model card](https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card)
* [MedSigLIP GitHub](https://github.com/Google-Health/medsiglip)
* [M3D repository](https://github.com/BAAI-DCAI/M3D)

## CT and MRI

* [CT-FM](https://github.com/project-lighter/CT-FM)
* [FlexiCT](https://github.com/ricklisz/FlexiCT)
* [Merlin](https://github.com/StanfordMIMI/Merlin)
* [Triad](https://github.com/wangshansong1/Triad)
* [NV-Segment-CTMR](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR)
* [MedSAM2](https://medsam2.github.io/)

## Pathology

* [H-Optimus](https://www.bioptimus.com/h-optimus)
* [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)
* [TITAN](https://github.com/mahmoodlab/TITAN)
* [TRIDENT](https://github.com/mahmoodlab/Trident)
* [OpenSlide](https://openslide.org/)
* [cuCIM](https://docs.rapids.ai/api/cucim/stable/)

## Data and interoperability

* [pydicom](https://pydicom.github.io/pydicom/stable/)
* [highdicom](https://highdicom.readthedocs.io/en/latest/)
* [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
* [RadGraph](https://github.com/Stanford-AIMI/radgraph)

A monthly watch for new medical foundation-model releases, checkpoint changes and license updates would help keep the registry current. Want me to schedule it?

[1]: https://monai.readthedocs.io/en/latest/bundle_intro.html?utm_source=chatgpt.com "Bundle — MONAI 1.6.0rc1 Documentation"
[2]: https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card?utm_source=chatgpt.com "MedSigLIP model card | Health AI Developer Foundations | Google for ..."
[3]: https://github.com/project-lighter/CT-FM?utm_source=chatgpt.com "GitHub - project-lighter/CT-FM: CT-FM: A 3D Image-Based Foundation ..."
[4]: https://github.com/wangshansong1/Triad?utm_source=chatgpt.com "GitHub - wangshansong1/Triad: code for \"Triad: Vision Foundation Model ..."
[5]: https://medsam2.github.io/?utm_source=chatgpt.com "MedSAM2: Segment Anything in 3D Medical Images and Videos"
[6]: https://www.bioptimus.com/h-optimus?utm_source=chatgpt.com "Foundational models for histology: H-Optimus"
[7]: https://developers.google.com/health-ai-developer-foundations/medgemma/model-card?utm_source=chatgpt.com "MedGemma 1.5 model card | Health AI Developer Foundations | Google for ..."
[8]: https://pydicom.github.io/pydicom/stable/index.html?utm_source=chatgpt.com "pydicom documentation — pydicom 3.0.2 documentation"
[9]: https://openslide.org/?utm_source=chatgpt.com "OpenSlide"
[10]: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes?utm_source=chatgpt.com "Transformers - Hugging Face"
[11]: https://github.com/MIC-DKFZ/nnUNet?utm_source=chatgpt.com "GitHub - MIC-DKFZ/nnUNet"
[12]: https://github.com/mahmoodlab/Trident?utm_source=chatgpt.com "GitHub - mahmoodlab/TRIDENT: Toolkit for large-scale whole-slide image ..."
[13]: https://github.com/ricklisz/FlexiCT/tree/main/flexi_ct?utm_source=chatgpt.com "FlexiCT/flexi_ct at main · ricklisz/FlexiCT · GitHub"
[14]: https://github.com/StanfordMIMI/Merlin/blob/main/pyproject.toml?utm_source=chatgpt.com "Merlin/pyproject.toml at main · StanfordMIMI/Merlin · GitHub"
[15]: https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/main/NV-Segment-CTMR?utm_source=chatgpt.com "NV-Segment-CTMR/NV-Segment-CTMR at main · NVIDIA-Medtech/NV ... - GitHub"
[16]: https://github.com/BAAI-DCAI/M3D?utm_source=chatgpt.com "GitHub - BAAI-DCAI/M3D: M3D: Advancing 3D Medical Image Analysis with ..."
[17]: https://github.com/prov-gigapath/prov-gigapath?utm_source=chatgpt.com "GitHub - prov-gigapath/prov-gigapath: Prov-GigaPath: A whole-slide ..."
[18]: https://github.com/mahmoodlab/TITAN?utm_source=chatgpt.com "GitHub - mahmoodlab/TITAN: Multimodal Whole Slide Foundation Model for ..."
[19]: https://huggingface.co/docs/peft/package_reference/lora?utm_source=chatgpt.com "LoRA · Hugging Face"
[20]: https://huggingface.co/docs/trl/use_model?utm_source=chatgpt.com "Use model after training - Hugging Face"
[21]: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation?utm_source=chatgpt.com "Performing gradient accumulation with Accelerate · Hugging Face"
[22]: https://pytorch.org/blog/activation-checkpointing-techniques/?utm_source=chatgpt.com "Current and New Activation Checkpointing Techniques in PyTorch"
[23]: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html?utm_source=chatgpt.com "torch.nn.functional.scaled_dot_product_attention"
[24]: https://monai-dev.readthedocs.io/en/stable/inferers.html?utm_source=chatgpt.com "Inference methods — MONAI 0 Documentation"
[25]: https://arxiv.org/abs/2604.05081?utm_source=chatgpt.com "MedGemma 1.5 Technical Report"
[26]: https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html?utm_source=chatgpt.com "AUROC — PyTorch-Metrics 1.9.0 documentation - Lightning"
[27]: https://github.com/Stanford-AIMI/radgraph?utm_source=chatgpt.com "GitHub - Stanford-AIMI/radgraph"
[28]: https://highdicom.readthedocs.io/en/latest/highdicom_and_pydicom.html?utm_source=chatgpt.com "Highdicom and Pydicom — highdicom 0.28.1 documentation"

