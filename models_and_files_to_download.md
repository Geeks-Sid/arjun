# Foundation Models, Frameworks, and Download Guide

This document provides a comprehensive analysis of all **foundation models, runtime frameworks, medical imaging libraries, and auxiliary tools** defined in `idea.md` and the `implementation_plan/` (Phases 00–18).

It includes their **roles, modalities, providers, official GitHub implementation links, Hugging Face weight URIs, license policies, and actionable step-by-step instructions for downloading and testing** on your Linux workstation (NVIDIA GPU with 48GB VRAM or Cloud TPU).

---

## 1. Quick Reference: V1 Foundation Model Roster

The framework uses a **single framework contract** with **modality-specific backbones** and PEFT/LoRA adaptation. Below is the full roster of **16 core models + 1 deferred model**:

| Model ID | Modality | Role | Provider / Organization | GitHub Implementation Link | Hugging Face Weights URI | License Status | MedFM Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`medsiglip`** | 2D Radiology | Preferred | Google Health AI | [Google-Health/medsiglip](https://github.com/Google-Health/medsiglip) | [google/medsiglip-448](https://huggingface.co/google/medsiglip-448) | HAI-DEF (Gated) | `blocked_unresolved` |
| **`rad-dino`** | 2D Radiology | Fallback | Microsoft | [microsoft/RAD-DINO](https://github.com/microsoft/RAD-DINO) | [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) | MIT | `approved_commercial` |
| **`ct-fm`** | 3D CT | Preferred | Project Lighter | [project-lighter/CT-FM](https://github.com/project-lighter/CT-FM) | [project-lighter/CT-FM](https://huggingface.co/project-lighter/CT-FM) | Academic / Open | `blocked_unresolved` |
| **`flexict-3d`** | 3D / 2D CT | Fallback | FlexiCT Authors | [ricklisz/FlexiCT](https://github.com/ricklisz/FlexiCT) | [ricklisz/FlexiCT](https://huggingface.co/ricklisz/FlexiCT) | Academic | `blocked_unresolved` |
| **`merlin`** | 3D CT VLM | Research | Stanford AIMI | [StanfordMIMI/Merlin](https://github.com/StanfordMIMI/Merlin) | [StanfordMIMI/Merlin](https://huggingface.co/StanfordMIMI) | Open Research | `blocked_unresolved` |
| **`triad`** | 3D MRI | Preferred | Triad Authors | [wangshansong1/Triad](https://github.com/wangshansong1/Triad) | Academic / GitHub Release | Academic | `blocked_unresolved` |
| **`nv-segment-ctmr`**| 3D CT/MRI Seg | Preferred | NVIDIA | [NVIDIA-Medtech/NV-Segment-CTMR](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR) | MONAI Bundle / NGC / HF | NVIDIA License | `blocked_unresolved` |
| **`medsam2`** | 3D/Video Seg | Preferred | Bo Wang Lab | [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2) | GitHub / HF Releases | Apache-2.0 / Custom | `blocked_unresolved` |
| **`m3d-lamed`** | 3D CT VLM | Research | BAAI-DCAI | [BAAI-DCAI/M3D](https://github.com/BAAI-DCAI/M3D) | [BAAI/M3D-LaMed](https://huggingface.co/BAAI) | Apache-2.0 / Open | `blocked_unresolved` |
| **`brainiac`** | 3D MRI | Deferred | BrainIAC Authors | Academic | Deferred post-v1 | Unreviewed | `deferred` |
| **`h-optimus-0`** | Pathology Tile | Preferred | Bioptimus | [Bioptimus/h-optimus](https://www.bioptimus.com/h-optimus) | [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | Apache-2.0 (Gated) | `pending_review` |
| **`gigapath-flash`** | Pathology WSI | Fallback | Microsoft / Providence | [prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath) | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) | Custom (Gated) | `blocked_unresolved` |
| **`titan`** | Pathology WSI | Optional | Mahmood Lab (Harvard/BWH) | [mahmoodlab/TITAN](https://github.com/mahmoodlab/TITAN) | [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) | Custom Research | `pending_review` |
| **`conch`** | Pathology Tile | Optional | Mahmood Lab (Harvard/BWH) | [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH) | [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) | Custom Research | `pending_review` |
| **`medgemma-1.5-4b`**| Generative VLM | Preferred | Google Health AI | [google-health/medgemma](https://github.com/google-health/medgemma) | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) | HAI-DEF (Gated) | `blocked_unresolved` |
| **`gemma-generic`** | Language | Optional | Google | [google/gemma](https://github.com/google/gemma_pytorch) | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | Gemma Terms | `pending_review` |
| **`qwen-generic`** | Language | Optional | Alibaba Qwen | [QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5) | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen) | Apache-2.0 | `pending_review` |

---

## 2. Detailed Breakdown of Foundation Models by Modality

### 2.1 2D Radiology Encoders

#### 1. MedSigLIP (`medsiglip`)
* **Role**: Primary 2D radiology image-text contrastive encoder (400M vision tower + 400M text tower, $448 \times 448$ input). Ideal for zero-shot classification, multi-label screening, and semantic image-text retrieval.
* **Provider**: Google (Health AI Developer Foundations)
* **GitHub Repository**: [https://github.com/Google-Health/medsiglip](https://github.com/Google-Health/medsiglip)
* **Hugging Face Model Weights**: [https://huggingface.co/google/medsiglip-448](https://huggingface.co/google/medsiglip-448)
* **Documentation & Model Card**: [https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card](https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card)
* **License**: Apache-2.0 (Code) / Health AI Developer Foundations Terms (Weights, Gated access).

#### 2. RAD-DINO (`rad-dino`)
* **Role**: Fallback 2D visual backbone trained via DINOv2 self-supervision on Chest X-Rays. Provides dense patch tokens and pooled embeddings for classification, segmentation, and visual bridging.
* **Provider**: Microsoft
* **GitHub Repository**: [https://github.com/microsoft/RAD-DINO](https://github.com/microsoft/RAD-DINO)
* **Hugging Face Model Weights**: [https://huggingface.co/microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino)
* **License**: MIT (Fully open for commercial use, modification, and redistribution).

---

### 2.2 3D Radiology (CT & MRI) Volumetric Foundation Models

#### 3. CT-FM (`ct-fm`)
* **Role**: Preferred 3D CT volumetric foundation model pretrained on large 3D CT collections. Provides feature maps, spatial tokens, and pooled representations for 3D segmentation, triage, and classification.
* **Provider**: Project Lighter
* **GitHub Repository**: [https://github.com/project-lighter/CT-FM](https://github.com/project-lighter/CT-FM)
* **Hugging Face Model Weights**: [https://huggingface.co/project-lighter/CT-FM](https://huggingface.co/project-lighter/CT-FM)
* **License**: Open Research.

#### 4. FlexiCT (`flexict-3d`)
* **Role**: Flexible CT foundation model family supporting 2D slice, 3D volume (`flexict_3d`), and 3D VLM (`flexict_3d_vlm`) input modes with patch-token and CLS outputs.
* **Provider**: FlexiCT Authors
* **GitHub Repository**: [https://github.com/ricklisz/FlexiCT](https://github.com/ricklisz/FlexiCT)
* **Hugging Face Model Weights**: [https://huggingface.co/ricklisz/FlexiCT](https://huggingface.co/ricklisz/FlexiCT)
* **License**: Open Research.

#### 5. Merlin (`merlin`)
* **Role**: Native 3D CT VLM built on MONAI, Transformers, and PEFT. Supports 3D CT embeddings, image-text contrastive alignment, and phenotype predictions.
* **Provider**: Stanford AIMI
* **GitHub Repository**: [https://github.com/StanfordMIMI/Merlin](https://github.com/StanfordMIMI/Merlin)
* **License**: Open Research.

#### 6. Triad (`triad`)
* **Role**: Preferred 3D MRI foundation model utilizing Swin-based architectures (MAE and SimMIM pretraining checkpoints) for multi-sequence 3D MRI classification, segmentation, and registration.
* **Provider**: Academic (wangshansong1 et al.)
* **GitHub Repository**: [https://github.com/wangshansong1/Triad](https://github.com/wangshansong1/Triad)
* **License**: Open Research.

#### 7. NV-Segment-CTMR (`nv-segment-ctmr`)
* **Role**: Multi-organ 3D CT and MRI automatic segmentation foundation model covering over 300 anatomical classes, distributed as a MONAI Bundle.
* **Provider**: NVIDIA
* **GitHub Repository / Bundle**: [https://github.com/NVIDIA-Medtech/NV-Segment-CTMR](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR)
* **MONAI Bundle Hub**: [https://docs.monai.io/en/stable/bundle_intro.html](https://docs.monai.io/en/stable/bundle_intro.html)
* **License**: NVIDIA Software License / Open MONAI Bundle.

#### 8. MedSAM2 (`medsam2`)
* **Role**: Promptable 3D medical image and video segmentation model incorporating SAM2 memory architectures for 3D CT and MRI volume segmentation.
* **Provider**: Bo Wang Lab (UHN / University of Toronto)
* **GitHub Repository**: [https://github.com/bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2)
* **Project Page**: [https://medsam2.github.io/](https://medsam2.github.io/)
* **License**: Apache-2.0 / Open Weights.

#### 9. M3D-CLIP & M3D-LaMed (`m3d-lamed`)
* **Role**: Comprehensive 3D Medical VLM family. `M3D-CLIP` handles 3D image-text retrieval and alignment; `M3D-LaMed` handles 3D VQA, 3D localization, and report generation.
* **Provider**: BAAI-DCAI
* **GitHub Repository**: [https://github.com/BAAI-DCAI/M3D](https://github.com/BAAI-DCAI/M3D)
* **Hugging Face Model Weights**: [https://huggingface.co/BAAI](https://huggingface.co/BAAI)
* **License**: Apache-2.0.

#### 10. BrainIAC (`brainiac`) — Deferred 3D MRI Candidate
* **Role**: 3D MRI brain imaging analysis and representation foundation model.
* **Provider**: BrainIAC Authors / Academic Research
* **Status in MedFM**: `deferred` post-v1 (`v1: false`).
* **Why Deferred**: In Phase 00 product requirements (`v1_scope.yaml`), 3D MRI preferred backbones are **Triad** (for classification/segmentation) and **NV-Segment-CTMR** (for multi-organ 3D segmentation). BrainIAC is registered as an optional post-v1 integration candidate once Triad and NV-Segment-CTMR are fully accepted.
* **License / Access**: Unreviewed / Academic repository pending Phase 05 post-v1 evaluation.

---

### 2.3 Pathology & Whole-Slide Image (WSI) Foundation Encoders

#### 11. H-Optimus-0 (`h-optimus-0`)
* **Role**: 1.1 Billion parameter vision encoder trained on hundreds of thousands of histology slides. Preferred tile-level encoder for digital pathology.
* **Provider**: Bioptimus
* **GitHub / Website**: [https://www.bioptimus.com/h-optimus](https://www.bioptimus.com/h-optimus)
* **Hugging Face Model Weights**: [https://huggingface.co/bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0)
* **License**: Apache-2.0 (Gated HF repo requiring terms agreement).

#### 12. Prov-GigaPath & GigaPath-Flash (`gigapath-flash`)
* **Role**: Whole-slide pathology foundation model architecture separating tile encoding (GigaPath Tile) from slide aggregation (GigaPath Slide / GigaPath-Flash).
* **Provider**: Microsoft / Providence Health
* **GitHub Repository**: [https://github.com/prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath)
* **Hugging Face Model Weights**: [https://huggingface.co/prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath)
* **License**: Custom Gated License.

#### 13. TITAN (`titan`)
* **Role**: Whole-slide multimodal encoder pretrained on whole-slide images aligned with pathology reports. Used for slide representations, retrieval, and report alignment.
* **Provider**: Mahmood Lab (Harvard / BWH)
* **GitHub Repository**: [https://github.com/mahmoodlab/TITAN](https://github.com/mahmoodlab/TITAN)
* **Hugging Face Model Weights**: [https://huggingface.co/MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN)
* **License**: Custom Non-Commercial Research License.

#### 14. CONCH (`conch`)
* **Role**: Vision-language foundation model for histology (tile retrieval, zero-shot classification, text-guided tile selection).
* **Provider**: Mahmood Lab (Harvard / BWH)
* **GitHub Repository**: [https://github.com/mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH)
* **Hugging Face Model Weights**: [https://huggingface.co/MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)
* **License**: Custom Non-Commercial Research License.

---

### 2.4 Generative Language & Vision-Language Foundation Models

#### 15. MedGemma 1.5 4B (`medgemma-1.5-4b`)
* **Role**: Primary generative medical VLM supporting multi-image 2D radiology, 3D CT/MRI slice sequences, and sampled WSI pathology tiles for VQA, report generation, and structured findings.
* **Provider**: Google Health AI
* **GitHub Repository**: [https://github.com/google-health/medgemma](https://github.com/google-health/medgemma)
* **Hugging Face Model Weights**: [https://huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
* **Documentation**: [https://developers.google.com/health-ai-developer-foundations/medgemma](https://developers.google.com/health-ai-developer-foundations/medgemma)
* **License**: Health AI Developer Foundations (HAI-DEF) Terms (Gated access).

#### 16. Generic Gemma / Qwen Baseline LMs (`gemma-generic`, `qwen-generic`)
* **Role**: Open-weights research comparison baselines pluggable behind the shared `LanguageModelAdapter` contract (e.g. Gemma 3 4B IT or Qwen 2.5 7B Instruct).
* **Providers**: Google / Alibaba Cloud
* **GitHub Repositories**:
  * Gemma: [https://github.com/google/gemma_pytorch](https://github.com/google/gemma_pytorch)
  * Qwen: [https://github.com/QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5)
* **Hugging Face Weights**:
  * Gemma: [https://huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
  * Qwen: [https://huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
* **Licenses**: Gemma Terms of Use / Apache-2.0.

---

### 2.5 Generic Adapters, Vision-Language Bridges, and MIL Aggregators ("and stuff")

Beyond individual pretrained foundation models, the framework architecture implements generic adapter fallbacks, vision-language bridges, and slide aggregators:

#### 1. Generic Vision Adapters (`GenericHFVisionAdapter`, `GenericTimmViTAdapter`)
* **Purpose**: Allow plugging any Hugging Face or `timm` Vision Transformer (e.g. DINOv2, ConvNeXt, Swin, EVA-02) into MedFM using standard `preprocess_spec()` and `encode()` protocols.
* **Upstream Libraries**: `transformers` ([GitHub](https://github.com/huggingface/transformers)) and `timm` ([GitHub](https://github.com/huggingface/pytorch-image-models)).

#### 2. Generic 3D MONAI Adapters (`GenericMONAI3DAdapter`)
* **Purpose**: Wrap MONAI 3D architectures (SwinUNETR, SegResNet, UNETR, DenseNet3D) for volumetric CT/MRI processing without writing custom wrappers per model.
* **Upstream Library**: `monai` ([GitHub](https://github.com/Project-MONAI/MONAI)).

#### 3. Vision-to-Language Bridges (`VisionLanguageBridge`)
* **Purpose**: Adapt visual spatial tokens $[B, N, D_v]$ from 2D, 3D, or WSI encoders into language model embedding dimension $[B, M, D_{lm}]$:
  * **Linear Projector**: $D_v \to D_{lm}$ mapping for smoke testing.
  * **Two-layer MLP Projector**: `Linear -> GELU -> LayerNorm -> Linear` default bridge.
  * **Perceiver Resampler**: Compresses arbitrary visual token counts into fixed query counts (32, 64, or 128 tokens) with coordinate encodings.
  * **Q-Former Bridge**: Query-based visual extraction for text-conditioned visual token selection.

#### 4. Slide Aggregators & Multiple Instance Learning (`GenericMILAggregator`)
* **Purpose**: Aggregate thousands of pathology tile embeddings $[T, D]$ into slide-level representations:
  * **Mean Pooling & Max Pooling**: Fast deterministic baselines.
  * **Attention MIL & Gated Attention MIL**: Learnable tile attention weighting.
  * **TransMIL / CLAM**: Transformer-based slide-level aggregators.
## 3. Core Frameworks, Libraries, and Tools

In addition to the foundation models, the framework relies on key upstream libraries specified in `pyproject.toml` and `references.md`:

### 3.1 Deep Learning Runtimes & Accelerators
* **PyTorch** (`torch`, `torchvision`): Pinned to version `2.9.0` (CUDA 12.8 wheels). [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
* **PyTorch/XLA** (`torch_xla`, `libtpu`): Pinned to version `2.9.0` with `libtpu 0.0.21` for Google Cloud TPU execution. [https://github.com/pytorch/xla](https://github.com/pytorch/xla)

### 3.2 Medical Imaging & Segmentation Foundations
* **MONAI**: Core medical transforms, sliding-window inferers, and 3D network architectures. [https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)
* **nnU-Net**: Self-configuring segmentation benchmark reference. [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

### 3.3 Hugging Face & PEFT Adaptation Stack
* **`transformers`**: Foundation model loading and tokenizers. [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
* **`peft`**: Parameter-Efficient Fine-Tuning (LoRA, RSLoRA, DoRA). [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
* **`accelerate`**: Mixed precision (BF16), gradient accumulation, and device placement. [https://github.com/huggingface/accelerate](https://github.com/huggingface/accelerate)
* **`bitsandbytes`**: 4-bit NormalFloat4 (NF4) QLoRA quantization on CUDA. [https://github.com/bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
* **`trl`**: Supervised Fine-Tuning (SFTTrainer) utilities. [https://github.com/huggingface/trl](https://github.com/huggingface/trl)

### 3.4 DICOM & Radiology I/O Tools
* **`pydicom`**: Low-level DICOM parsing and header extraction. [https://github.com/pydicom/pydicom](https://github.com/pydicom/pydicom)
* **`highdicom`**: DICOM computational imaging and derived object generation. [https://github.com/highdicom/highdicom](https://github.com/highdicom/highdicom)
* **`nibabel`**: NIfTI volumetric data I/O. [https://github.com/nipy/nibabel](https://github.com/nipy/nibabel)
* **`SimpleITK`**: ITK-based medical image registration and resampling. [https://github.com/SimpleITK/SimpleITK](https://github.com/SimpleITK/SimpleITK)

### 3.5 Pathology & WSI Processing Tools
* **`OpenSlide` (`openslide-python`)**: Multi-resolution gigapixel WSI slide reading. [https://github.com/openslide/openslide-python](https://github.com/openslide/openslide-python)
* **`tiffslide`**: Pure-Python TIFF/SVS slide reader fallback. [https://github.com/buelowp/tiffslide](https://github.com/buelowp/tiffslide)
* **`cuCIM`**: GPU-accelerated multidimensional image I/O (RAPIDS). [https://github.com/rapidsai/cucim](https://github.com/rapidsai/cucim)
* **`TRIDENT`**: Whole-slide tissue segmentation and patch coordinate extraction pipeline. [https://github.com/mahmoodlab/Trident](https://github.com/mahmoodlab/Trident)

### 3.6 Evaluation & Clinical Metrics
* **`torchmetrics`**: Standard classification, segmentation, and generation metrics. [https://github.com/Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics)
* **`RadGraph`**: Entity and relation extraction for radiology report evaluation. [https://github.com/Stanford-AIMI/radgraph](https://github.com/Stanford-AIMI/radgraph)

---

## 4. Downloading and Setup Guide for Testing

### Step 1: Initialize the Local Environment
Run the environment setup to install all core dependencies, medical extras, and dev tools using `uv`:

```bash
# Clone the framework repository (if not already local)
cd /home/siddhesh/Work/Personal/arjun

# Install standard runtime + medical + HF/PEFT extras + CUDA tools
make install-dev

# Run environment diagnostic to verify GPU and library versions
python -m medfm.tools.doctor --backend auto
```

### Step 2: Hugging Face CLI Authentication
Many foundation models (e.g., `h-optimus-0`, `medsiglip-448`, `medgemma-1.5-4b-it`, `prov-gigapath`, `CONCH`, `TITAN`) are **gated on Hugging Face**. You must log in with an authenticated Hugging Face User Access Token:

```bash
# Install Hugging Face CLI
uv pip install huggingface_hub

# Authenticate with your Hugging Face Token (requires read permissions)
huggingface-cli login
```

*Note: Before downloading gated models, visit their respective Hugging Face repo pages listed in Section 2 and click "Accept Terms & Conditions".*

### Step 3: Downloading Recommended Base Models

#### A. Approved Open Models (Downloadable Immediately)

**1. RAD-DINO (2D Radiology - MIT License):**
```bash
# Download weights directly via Hugging Face CLI
huggingface-cli download microsoft/rad-dino --local-dir ~/.cache/huggingface/hub/models--microsoft--rad-dino
```

**2. H-Optimus-0 (Pathology Tile - Apache-2.0, Gated):**
```bash
huggingface-cli download bioptimus/H-optimus-0 --local-dir ~/.cache/huggingface/hub/models--bioptimus--H-optimus-0
```

**3. Qwen 2.5 7B Instruct (Generic LM Baseline - Apache-2.0):**
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
```

#### B. Gated Medical Foundation Models (Requires Accepted HF Terms)

**4. MedSigLIP 448 (2D Radiology - HAI-DEF Gated):**
```bash
huggingface-cli download google/medsiglip-448 --local-dir ~/.cache/huggingface/hub/models--google--medsiglip-448
```

**5. MedGemma 1.5 4B IT (Generative Medical VLM - HAI-DEF Gated):**
```bash
huggingface-cli download google/medgemma-1.5-4b-it --local-dir ~/.cache/huggingface/hub/models--google--medgemma-1.5-4b-it
```

**6. Prov-GigaPath (Pathology Slide/Tile - Gated):**
```bash
huggingface-cli download prov-gigapath/prov-gigapath --local-dir ~/.cache/huggingface/hub/models--prov-gigapath--prov-gigapath
```

#### C. Cloning Upstream Implementation Repositories for Direct Testing

```bash
# Create a workspace directory for external repositories
mkdir -p external_repos && cd external_repos

# 2D & Medical VLMs
git clone https://github.com/Google-Health/medsiglip.git
git clone https://github.com/microsoft/RAD-DINO.git
git clone https://github.com/google-health/medgemma.git

# 3D Radiology (CT / MRI / Segmentation)
git clone https://github.com/project-lighter/CT-FM.git
git clone https://github.com/ricklisz/FlexiCT.git
git clone https://github.com/StanfordMIMI/Merlin.git
git clone https://github.com/wangshansong1/Triad.git
git clone https://github.com/bowang-lab/MedSAM2.git
git clone https://github.com/BAAI-DCAI/M3D.git

# Pathology
git clone https://github.com/prov-gigapath/prov-gigapath.git
git clone https://github.com/mahmoodlab/TITAN.git
git clone https://github.com/mahmoodlab/Trident.git

cd ..
```

---

## 5. MedFM Framework Integration & Testing

### Using the MedFM Model Registry CLI
The framework includes built-in CLI commands in `medfm.cli.models` to list, inspect, download, and smoke-test registered models:

```bash
# 1. List all registered models and their capability status
python -m medfm.cli.models list

# 2. Inspect exact specifications and memory requirements for a model
python -m medfm.cli.models show rad-dino

# 3. Estimate memory consumption under QLoRA / BF16 modes
python -m medfm.cli.models estimate-memory rad-dino

# 4. Run a local smoke test on synthetic inputs (CPU / CUDA)
python -m medfm.cli.models smoke rad-dino
```

### Running Test Suites
To verify that all core contracts, data readers, adapters, and trainer pipelines are functional:

```bash
# Run CPU unit & contract tests (no model weights needed)
make test

# Run GPU smoke & execution tests on your 48GB workstation GPU
make test-gpu

# Run overall phase acceptance validation
python -m medfm.tools.validate_phase --phase 01
---

## 6. Additional Medical & CT Foundation Models in Literature (Post-v1 & Extended Candidates)

While the core MedFM v1 roster includes **16 primary models + 1 deferred model** (`ct-fm`, `flexict-3d`, `merlin`, `m3d-lamed`, `nv-segment-ctmr`, `medsam2`, `brainiac`, etc.), the broader medical AI literature contains several additional notable foundation models. All of these can be plugged into MedFM via the generic adapter layer (`GenericMONAI3DAdapter`, `GenericHFVisionAdapter`, `GenericHFCausalLMAdapter`).

### 6.1 Additional 3D CT Foundation Models

1. **TotalSegmentator (v1 & v2)**
   * **Role**: Anatomical 3D CT multi-organ segmentation foundation model based on nnU-Net, capable of segmenting 117+ anatomical structures (organs, bones, muscles, vessels) in CT.
   * **GitHub**: [https://github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
   * **Integration**: Loadable via `GenericMONAI3DAdapter` or native TotalSegmentator PyPI package.

2. **SwinUNETR / SegResNet Pretrained Encoders (BTCV / MONAI Model Zoo)**
   * **Role**: Self-supervised 3D CT transformer & CNN backbones pretrained on multi-site 3D CT datasets (Beyond the Cranial Vault) for abdominal organ and tumor segmentation.
   * **GitHub**: [https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR)
   * **MONAI Model Zoo**: [https://monai.io/model-zoo.html](https://monai.io/model-zoo.html)

3. **AbdomenCT-1K / AbdomenCT-12K Encoders**
   * **Role**: Large-scale 3D CT foundation models pretrained on 1,000 to 12,000 abdominal CT scans for organ segmentation and lesion detection.
   * **GitHub**: [https://github.com/JunMa11/AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)

4. **STIP (Spatio-Temporal Image Pre-training for 3D CT)**
   * **Role**: Self-supervised representation learning backbone specifically designed for 3D CT scans and longitudinal CT slice sequences.
   * **GitHub**: [https://github.com/v3ntus/STIP](https://github.com/v3ntus/STIP)

5. **RadImageNet / RadImageNet-3D**
   * **Role**: Open radiologic foundation model pretrained on 1.35 million radiologic images (CT, MRI, Ultrasound) across 11 anatomical regions.
   * **GitHub**: [https://github.com/RadImageNet/RadImageNet](https://github.com/RadImageNet/RadImageNet)

---

### 6.2 Additional Pathology Foundation Models

1. **UNI (Mahmood Lab)**
   * **Role**: General-purpose pathology vision transformer foundation model pretrained on 100,000+ histology WSIs across 20+ tissue types.
   * **GitHub**: [https://github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)
   * **Hugging Face**: [https://huggingface.co/MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)

2. **Virchow & Virchow 2 (Paige AI)**
   * **Role**: 632M parameter open-weights pathology vision transformer trained on 1.5M+ whole-slide images.
   * **GitHub**: [https://github.com/paigeai/Virchow](https://github.com/paigeai/Virchow)
   * **Hugging Face**: [https://huggingface.co/paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2)

3. **Phikon & Phikon-v2 (Owkin)**
   * **Role**: Open-weights pathology ViT foundation models trained on TCGA and proprietary tissue banks.
   * **GitHub**: [https://github.com/owkin/phikon](https://github.com/owkin/phikon)
   * **Hugging Face**: [https://huggingface.co/owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2)

4. **PLIP (Pathology Language-Image Pre-training)**
   * **Role**: Open pathology vision-language model trained on Twitter/Open-access pathology image-text pairs.
   * **GitHub**: [https://github.com/PathologyAI/PLIP](https://github.com/PathologyAI/PLIP)


### 6.3 Additional 2D Radiology & Medical Vision-Language Models

1. **BiomedCLIP (Microsoft Research)**
   * **Role**: Open-source medical vision-language contrastive model trained on 15M image-text pairs from PMC-OA.
   * **GitHub**: [https://github.com/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://github.com/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
   * **Hugging Face**: [https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

2. **CheXzero (Stanford AIMI)**
   * **Role**: Zero-shot Chest X-Ray interpretation foundation model requiring no manual labels.
   * **GitHub**: [https://github.com/stanfordmlgroup/CheXzero](https://github.com/stanfordmlgroup/CheXzero)

3. **BioViL & BioViL-T (Microsoft Research)**
   * **Role**: Specialized vision-language model for temporal chest radiograph tracking and report generation.
   * **GitHub**: [https://github.com/microsoft/hi-ml](https://github.com/microsoft/hi-ml)

4. **LLaVA-Med / LLaVA-Rad**
   * **Role**: Instruction-tuned biomedical vision-language assistant models based on LLaVA.
   * **GitHub**: [https://github.com/microsoft/LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
```

---

## Summary Recommendation for Immediate Testing

To quickly establish working baselines:
1. **Start with `rad-dino`**: Fully approved under MIT, immediate download, no gated access delays, lightweight 2D radiology testing.
2. **Accept HF Terms for `medsiglip-448` & `medgemma-1.5-4b-it`**: Gives access to Google's preferred 2D contrastive encoder and 4B generative VLM.
3. **Accept HF Terms for `bioptimus/H-optimus-0`**: Gives access to the preferred 1.1B pathology tile encoder under Apache-2.0.
4. **Use local synthetic tests (`make test`)**: Verify all MONAI transforms, PEFT/LoRA modules, and batch collators without requiring full dataset downloads.
