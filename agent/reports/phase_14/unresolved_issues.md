# Phase 14 unresolved issues

- TPU PJRT/BF16 execution was not run because this workstation has no
  `torch_xla` runtime or TPU device. The published TPU profiles use fixed
  shape buckets and explicit `tpu_status` values; they are not hardware
  acceptance evidence.
- CUDA execution, CUDA QLoRA/NF4, multi-device distribution, and peak VRAM
  measurements were not run in this CPU-only acceptance pass.
- Approved production CT/MRI/VQA/query datasets and released upstream
  checkpoints were not available. Offline-tiny weights and synthetic tensors
  remain deterministic contract fixtures only.
- External-site calibration, original-space representative visualizations,
  clinical reader review, and clinical safety evidence remain required before
  any deployment decision.
- The native 3D VLM stage-4 region/box/mask output fields are declared and
  audited as an opt-in contract, but no clinical region annotation benchmark
  was executed in this phase.

- A repository-wide `python -m pytest -q` run remains blocked outside this phase
  by the missing `safetensors` dependency and duplicate top-level test module
  names in older phase directories; the Phase 12–14 regression is green.