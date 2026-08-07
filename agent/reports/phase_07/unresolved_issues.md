# Unresolved and explicitly gated items

- CT-FM, FlexiCT, Triad, NV-Segment-CTMR, MedSAM2, Merlin, and M3D upstream checkpoint identities remain blocked by unresolved license/checkpoint records in `model_registry/licenses.yaml`. The adapters provide offline contract implementations and do not imply upstream weights are loadable.
- MONAI bundle import and MedSAM2 upstream sequential memory operators are recorded as gated/custom limitations. The generic pure-PyTorch fallback is the tested baseline.
- CUDA and TPU real-model peak VRAM/XLA compilation reports require the corresponding protected hardware. Fixed-shape TPU smoke configuration and host-side window batching are implemented; hardware evidence is not fabricated.
- Native language generation for Merlin/M3D-LaMed and FlexiCT VLM bridging remain separate gates; visual features are exposed, generation is rejected until a reviewed checkpoint and memory profile are supplied.
