# transfer_plan/INDEX.md — full file → checklist map

Every `medfm/**/*.py` source file is accounted for. Use this to dispatch parallel agents:
each row names the checklist that owns the file and its wave.

## evaluation/
| File | Checklist | Wave |
|---|---|---|
| evaluation/advanced.py | evaluation/advanced.md | 0 |
| evaluation/metrics.py | evaluation/metrics.md | 1 |
| evaluation/calibration.py | evaluation/calibration.md | 1 |
| evaluation/uncertainty.py | evaluation/uncertainty.md | 1 |
| evaluation/distributed.py | evaluation/distributed.md | 1 |
| evaluation/specialized.py | evaluation/specialized.md | 1 |
| evaluation/ablation.py | README §5 (keep — orchestration) | 2 |
| evaluation/artifacts.py | README §5 (keep — IO over metrics) | 2 |
| evaluation/report.py | README §5 (keep — report IO) | 2 |
| evaluation/human_review.py | README §5 (keep — review workflow) | 2 |
| evaluation/schemas.py | README §5 (keep — pydantic schemas) | 2 |

## inference/
| File | Checklist | Wave |
|---|---|---|
| inference/sliding_window.py | inference/sliding_window.md | 0 |
| inference/generation.py | inference/generation.md | 1 |
| inference/pipeline.py | inference/pipeline.md | 1 |
| inference/bundle.py | inference/bundle.md | 1 |
| inference/server.py | README §5 (keep — stdlib HTTP/LRU) | 2 |
| inference/audit.py | README §5 (keep — audit log) | 2 |
| inference/schemas.py | README §5 (keep — pydantic) | 2 |
| inference/errors.py | README §5 (keep) | 2 |
| inference/export_nifti.py | README §5 (keep — nibabel already) | 2 |
| inference/export_dicom.py | README §5 (keep — highdicom already) | 2 |

## data/
| File | Checklist | Wave |
|---|---|---|
| data/transforms/radiology2d.py | data/transforms/radiology2d.md | 0 |
| data/transforms/spatial3d.py | data/transforms/spatial3d.md | 0 |
| data/transforms/ct.py | data/transforms/ct.md | 0 |
| data/transforms/mri.py | data/transforms/mri.md | 0 |
| data/transforms/pathology.py | data/transforms/pathology.md | 0 |
| data/transforms/{base,pipeline,specs,timing}.py | README §5 (keep — invertible-history contract) | 2 |
| data/readers/{base,dicom,radiology,pathology}.py | data/readers.md | 2 |
| data/samplers/patches.py | data/samplers/patches.md | 1 |
| data/samplers/distributed.py | data/samplers/distributed.md → collators-caching-fingerprint.md | 2 |
| data/collators/* | data/collators-caching-fingerprint.md | 2 |
| data/caching/* | data/collators-caching-fingerprint.md | 2 |
| data/fingerprint.py | data/collators-caching-fingerprint.md | 2 |
| data/manifests/* | data/collators-caching-fingerprint.md | 2 |
| data/textprep/* | data/collators-caching-fingerprint.md | 2 |
| data/splits.py | data/splits.md | 2 |

## models/
| File | Checklist | Wave |
|---|---|---|
| models/decoders/unet.py | models/decoders/unet.md | 0 |
| models/decoders/fpn.py | models/decoders/fpn.md | 0 |
| models/decoders/{base,masks,language,segmentation}.py | models/decoders-heads-bridges-base-keep.md | 1 |
| models/heads/localization.py | models/heads/localization.md | 0 |
| models/heads/pooling.py | models/heads/pooling.md | 0 |
| models/heads/retrieval.py | models/heads/retrieval.md | 0 |
| models/heads/{classification,losses}.py | models/decoders-heads-bridges-base-keep.md | 1 |
| models/bridges/resampler.py | models/bridges/resampler.md | 0 |
| models/bridges/{base,coordinates,placement,training}.py | models/decoders-heads-bridges-base-keep.md | 1 |
| models/pathology/aggregation.py | models/pathology/aggregation.md | 0 |
| models/pathology/selection.py | models/pathology/selection.md | 1 |
| models/pathology/{aggregators,selectors,stores,distributed,encoders,adapters,pipeline}.py | models/decoders-heads-bridges-base-keep.md | 1 |
| models/visual/{base,hf_generic,hoptimus0,medsiglip,raddino,medgemma_vision,ct_fm,triad,research_3d,native_tasks}.py | models/visual/adapters.md | 2 |
| models/visual/native_3d.py | models/visual/native_3d.md | 0 |
| models/language/* | models/language.md | 2 |

## peft/ · tasks/ · training/ · other
| File | Checklist | Wave |
|---|---|---|
| peft/lora.py | peft/lora.md | 0 (cautious) |
| peft/{resolver,checkpoint,config,quantization,errors}.py | README §5 (keep — peft wraps bnb/transformers already) | 2 |
| tasks/losses.py | tasks/losses.md | 0 |
| tasks/{classification,segmentation,retrieval,localization,boxes,alignment,generation,language_segmentation,multitask,reductions,structured,structured_generation,schemas,base}.py | tasks/keep-cluster.md | 1 |
| training/optimizer.py | training/optimizer.md | 0 |
| training/{memory,checkpoint,distributed,data,steps,trainer,pipeline,evaluation,tracking,run_metadata,config}.py | training/keep-cluster.md + training/backend.md | 2 |
| core/* | core/serialization.md + README §5 (contract layer) | 2 |
| registry/*, cli/*, tools/*, recipes/* | README §5 (keep — orchestration/glue) | 2 |

## Parallelization rules
- Run **all Wave-0** checklists in parallel (independent leaves).
- **Wave-1** after their Wave-0 upstreams merge: `evaluation/metrics.md`,
  `evaluation/{calibration,uncertainty,distributed,specialized}.md`, `inference/{pipeline,generation}.md`,
  `tasks/keep-cluster.md`, `data/samplers/patches.md`, `models/pathology/selection.md`,
  `models/visual/native_3d.md` (after `inference/sliding_window.md`), `models/decoders-heads-bridges-base-keep.md`.
- **Wave-2** is verification-only (keep by design): run full suite once.
