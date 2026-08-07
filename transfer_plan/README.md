# medfm → Open-Source / Library Transfer Plan

Goal: **replace hand-rolled implementations with mature open-source libraries** wherever a
library already covers the exact behavior the framework needs, in a way that is **testable**
and preserves the framework's contracts. This directory is the coordination point: one
checklist per source file so the migration can run as **parallel agent work units**.

Scope note: this is a *plan*. The actual code changes are backfilled by agents executing
each checklist. A file is "done" when its checklist items are ticked, the file's tests pass,
and `make lint && make typecheck` stay green.

---

## 1. Ground rules (derived from the request + repo contracts)

1. **Only transfer when the library truly covers it.** If the library cannot express the
   exact behavior (empty-mask semantics, physical-space distances, TPU static-shape
   buckets, invertible transform history, accelerator-neutral contract types), keep the
   hand-rolled code and say so in the checklist — marking it `keep`.
2. **No forced dtype casts.** Do not cast `torch.float16`/`bfloat16`/`int64`→`float32`
   merely to make a library call type-check. A transfer is only eligible when the library
   natively accepts the tensor dtype in play (same dtype portability). If a candidate needs
   a semantics-changing cast, it gets `keep` (or a documented `partial` with an explicit
   wrapper that already handles dtype).
3. **Conservative with training/optimizer libraries.** Do not swap the trainer/optimizer
   orchestration for a trainer framework unless the library is extremely mature *and* the
   value is clear. Leaf math (losses, metrics, bleeding kernels) is fair game.
4. **Testability is mandatory.** Every transfer must be backed by an existing test or a new
   parity test. The canonical example is **Hausdorff distance**: `medfm/evaluation/advanced.py`
   computes HD95/ASSD/surface-Dice by hand and MONAI has
   `monai.metrics.HausdorffDistanceMetric`/`SurfaceDistanceMetric`/`SurfaceDiceMetric` — the
   plan requires a number-for-number parity test before the swap (see
   `evaluation/advanced.md`). The pad of such a test must match the repo's empty-mask
   conventions, which **MONAI does not** (see §2).
5. **Preserve public signatures and contract types.** `MetricValue`, `EncoderOutput`,
   `LossOutput`, `MedicalBatch`, `TransformRecord` inversion history, `BucketId`/buckets are
   load-bearing. Transfers happen inside the body; public faces stay stable.
6. **Optional extras stay optional.** `medical` (monai, nibabel, pydicom, SimpleITK,
   scikit-image), `pathology` (openslide, tiffslide, h5py, zarr, pyarrow), `hf`
   (transformers, peft, timm), `cuda` (bitsandbytes), `tpu` (torch_xla). A transfer is only
   allowed where that dependency is already in the file's guarded-import surface; adding a
   new **mandatory** dependency requires an ADR and is out of scope here. We prefer
   libraries already installed in `.venv` (see §3) with **zero new deps**.
7. **External reference repos may be borrowed** (copy file, keep license header) where no
   installed library covers the gap: `external_repos/` contains CONCH, TITAN, UNI,
   prov-gigapath, Triad/Trident, CT-FM, medgemma, medsiglip, M3D, MedSAM2, TotalSegmentator,
   Merlin, etc. Reuse must respect each repo's LICENSE (see
   `docs/licensing_policy.md`): CONCH/UNI/Trident are CC-BY-NC-ND-4.0 (research only),
   prov-gigapath/CT-FM/M3D/MedSAM2/merlin are Apache-2.0/MIT (commercial-safe).

---

## 2. Verified library surface (installed in `.venv`)

All API names below were **import-verified** against the pinned versions
(torch 2.9.0, torchvision 0.24.0, monai 1.6.0, transformers 5.14.1, peft 0.20.0,
accelerate 1.14.0, timm 1.0.28, scikit-image 0.25, scipy 1.15):

| Domain | Library API | Notes |
|---|---|---|
| Sliding-window 3D inference | `monai.inferers.sliding_window_inference` | roi_size, sw_batch_size, overlap, `mode="gaussian"` (default sigma_scale 0.125 matches ours), padding_mode, ndim any |
| Gaussian importance map | `monai.utils.gaussian_1d` | separable, window_shape→[1,1,…], positive |
| Hausdorff / ASSD / surface-Dice | `monai.metrics.HausdorffDistanceMetric`, `SurfaceDistanceMetric`, `SurfaceDiceMetric` | **returns NaN on empty masks**; `include_background`, `percentile`, `distance_metric`, `reduction`, `class_thresholds` |
| Dice / IoU | `monai.metrics.DiceMetric` | `include_background`, `reduction`, `ignore_empty` |
| ROC / AUC | `monai.metrics.ROCAUCMetric` / `monai.metrics.auc` (available via `monai` metrics; `sklearn`/`torchmetrics` **not installed**) | ROC-AUC, PR-AUC available |
| Losses | `monai.losses.DiceLoss`, `DiceCELoss`, `FocalLoss` (`gamma`/`alpha`/`use_softmax`), `GeneralizedDiceLoss`, `TverskyLoss` (`alpha`/`beta`), `ContrastiveLoss`, `DeepSupervisionLoss` | torch-native, dtype-agnostic (assumes logits/one-hot) |
| 3D spatial transforms | `monai.transforms` `Orientationd`/`Orientation`, `Spacing`/`Spacingd`, `Resized`/`ScaleIntensityRanged`, `NormalizeIntensityd`, `CropForegroundd` (`select_fn`), `RandGaussianNoised`/`RandAffined` | dict- or array-based; **no invertible-history equivalent** — our `TransformRecord` bookkeeping stays custom |
| 2D transforms | `torchvision.transforms.functional` (`resize`, `affine`, `rotate`, `crop`, `hflip`, `vflip`, `adjust_*`, `gaussian_blur`) + `torchvision.transforms` v1 (no v2 in this pin) | PIL/tensor, dtype-preserving for float32 tensors |
| Networks | `monai.networks.nets.UNet`, `BasicUNet`, `FlexibleUNet`, `DynUNet`, `monai.networks.blocks` (`Convolution`, `UpSample`, `MLPBlock`, `TransformerBlock`) | torch-native; consume raw volumes, not EncoderOutput pyramids |
| FPN | `torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork` | **2D only**; 3D FPN has no library equivalent |
| Box ops | `torchvision.ops.box_iou`, `generalized_box_iou`, `complete_box_iou`, `distance_box_iou`, `ciou_loss`, `diou_loss`, `giou_loss`, `nms`, `roi_align` | **2D (N,4) only**; 3D boxes need hand-rolled math |
| LoRA | `peft.LoraConfig`, `peft.get_peft_model`, `peft.PeftModel`, `peft.tuners.inject_adapter_in_model` | mature; **CUDA/TPU-note**: our hand-rolled LoRA deliberately avoids bitsandbytes on TPU — verify before adoption |
| Quantization | `bitsandbytes` (`cuda` extra), `transformers.BitsAndBytesConfig`, `peft.prepare_model_for_kbit_training` | CUDA-only; guarded |
| WSI | `openslide`, `tiffslide`, `cucim` (guarded), `h5py`/`zarr` | readers already wrap these |
| Misc | `scipy.ndimage` (distance_transform_edt, zoom, label), `skimage.filters.threshold_otsu`, `skimage.transform.resize` | already installed |

**Not installed → new dependency, out of scope unless approved:** scikit-learn, torchmetrics.

---

## 3. Dependency / parallelization model

The migration work is **one agent per file** (or per small cluster of files). Parallelism is
bounded by a *must-run-before* relation, not by the transfer itself. The layered layout:

```
Wave 0 — leaves (no internal package deps; fully parallel, high confidence):
  evaluation/advanced.py            (metric kernels: HD95/ASSD/surface-Dice/Dice/AUC)
  inference/sliding_window.py       (sliding-window + Gaussian blending)
  data/transforms/{radiology2d,spatial3d,ct,mri,pathology}.py
  models/decoders/{unet,fpn}.py
  models/heads/{localization,pooling,retrieval}.py   (box math, pooling, contrastive)
  models/pathology/{aggregation,selection}.py
  models/bridges/resampler.py
  tasks/losses.py                   (loss kernels)
  peft/lora.py                      (LoRA; high caution)
  training/optimizer.py             (scheduler math only)

Wave 1 — depends on Wave 0 symbols (run after their upstream merges):
  evaluation/metrics.py             (facade → advanced)
  evaluation/calibration.py, uncertainty.py, specialized.py, distributed.py
  models/heads/losses.py            (re-exports tasks.losses)
  tasks/{classification,segmentation,retrieval,localization,boxes,alignment}.py
  models/decoders/segmentation.py   (re-export)
  models/pathology/{aggregators,selectors,stores,encoders}.py  (re-export)
  inference/{pipeline,generation}.py
  data/samplers/patches.py

Wave 2 — orchestration (keep; only verify no breakage):
  training/{trainer,steps,backend,memory,checkpoint,distributed,data,tracking}.py
  models/language/*, models/visual/*, recipes/*, registry/*, cli/*, tools/*, core/*
```

Files in Wave 2 are **`keep`** by design (orchestration/contract/glue) — no agent needed
beyond running the suite.

Work units are named `transfer_plan/<package>/<file>.md`.

---

## 4. Verification protocol (what "done" means for each checklist)

1. Apply the transfer in the target file only; keep the public surface.
2. Ensure zero new mandatory dependencies (or file an ADR if unavoidable).
3. Run the file's focused tests + the parity tests the checklist calls out.
4. Run `ruff check` and `mypy` (strict) on the file.
5. Final per-wave gate: `make lint && make typecheck && make test`.
6. Tick every `- [ ]` in the checklist as completed; add a `## Result` section
   summarizing what was transferred vs kept and any numeric drift found by parity tests.

---

## 5. Master inventory

Legend: `transfer` = library replaces body; `partial` = library covers most, custom glue
remains (spelled out per function); `keep` = hand-rolled essential (no library equivalent,
contract semantics, or requires a new dep out of scope).

| File | Verdict | Library target |
|---|---|---|
| evaluation/advanced.py | partial | monai.metrics, monai.utils |
| evaluation/metrics.py | transfer (facade) | advanced.py |
| evaluation/calibration.py | keep | (no installed lib for ECE/Brier; torchmetrics not installed) |
| evaluation/uncertainty.py | keep | |
| evaluation/distributed.py | keep | |
| evaluation/specialized.py | keep | |
| inference/sliding_window.py | partial | monai.inferers.sliding_window_inference |
| inference/generation.py | keep | (already wraps transformers.generate) |
| inference/pipeline.py | keep | |
| inference/bundle.py, audit.py, server.py, schemas.py, errors.py | keep | |
| inference/export_nifti.py | keep (already nibabel) | |
| inference/export_dicom.py | keep (already highdicom) | |
| data/transforms/radiology2d.py | partial | torchvision.transforms.functional |
| data/transforms/spatial3d.py | partial | monai.transforms (orientation/spacing kernels) |
| data/transforms/ct.py | partial | monai ScaleIntensityRanged / torch |
| data/transforms/mri.py | partial | monai NormalizeIntensityd / SimpleITK |
| data/transforms/pathology.py | partial | skimage.filters.threshold_otsu, skimage.transform |
| data/transforms/base.py, pipeline.py, specs.py, timing.py | keep | (invertible-history contract) |
| data/readers/{base,dicom,radiology,pathology}.py | keep | (already wrap pydicom/nibabel/SimpleITK/openslide/tiffslide) |
| data/collators/* | keep | (TPU static-shape buckets) |
| data/samplers/{distributed,patches}.py | keep | |
| data/caching/* | keep | (safetensors already) |
| data/manifests/*, splits.py, fingerprint.py | keep | (pyarrow/pandas/hashlib already) |
| data/textprep/* | keep | |
| models/decoders/unet.py | partial | monai networks blocks |
| models/decoders/fpn.py | partial | torchvision FPN (2D only) |
| models/decoders/{base,masks,language,segmentation}.py | keep | |
| models/heads/localization.py | partial | torchvision.ops (2D box) |
| models/heads/pooling.py | keep | |
| models/heads/retrieval.py | keep | (contrastive is CLIP-style; monai ContrastiveLoss differs) |
| models/heads/{classification,losses}.py | keep | |
| models/bridges/resampler.py | keep | (custom Perceiver; transformers has no drop-in resampler) |
| models/bridges/* | keep | |
| models/pathology/aggregation.py | keep | (attention-MIL hand-rolled; see external_repos note) |
| models/pathology/selection.py | keep | |
| models/pathology/* | keep | |
| models/visual/* | keep | (already wrap timm/transformers/native) |
| models/language/* | keep | (already wrap transformers) |
| peft/lora.py | partial (cautious) | peft (mature) — see checklist caveats |
| peft/{resolver,checkpoint,config,quantization}.py | keep | |
| tasks/losses.py | partial | monai.losses |
| tasks/*.py | keep | |
| training/optimizer.py | partial (schedule) | torch.optim.lr_scheduler |
| training/* | keep | |
| recipes/*, registry/*, cli/*, tools/*, core/* | keep | |

See per-file checklists below for per-function detail, parity-test requirements, and
wave placement.
