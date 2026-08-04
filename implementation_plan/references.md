# Primary Implementation References

Use primary documentation, model cards, and upstream repositories when implementing a phase. Pin the exact dependency/model revision in registry and run artifacts; these links identify the authority, not a floating runtime dependency.

## Framework and accelerator references

- [PyTorch](https://docs.pytorch.org/docs/stable/)
- [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch distributed checkpointing](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [PyTorch/XLA](https://docs.pytorch.org/xla/master/)
- [PyTorch/XLA SPMD](https://docs.pytorch.org/xla/master/spmd.html)
- [PyTorch/XLA profiling](https://docs.pytorch.org/xla/master/learn/xla-profiling.html)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)
- [Accelerate TPU training](https://huggingface.co/docs/accelerate/basic_tutorials/tpu)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)
- [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)
- [Transformers bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [Hugging Face TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [MONAI](https://docs.monai.io/en/stable/)
- [MONAI Bundles](https://docs.monai.io/en/stable/bundle_intro.html)
- [MONAI inferers and sliding-window inference](https://docs.monai.io/en/stable/inferers.html)

## Data and interoperability references

- [pydicom](https://pydicom.github.io/pydicom/stable/)
- [highdicom](https://highdicom.readthedocs.io/en/latest/)
- [NiBabel](https://nipy.org/nibabel/)
- [OpenSlide](https://openslide.org/)
- [cuCIM](https://docs.rapids.ai/api/cucim/stable/)
- [TRIDENT](https://github.com/mahmoodlab/Trident)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

## Medical image and VLM references

- [MedSigLIP model card](https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card)
- [MedSigLIP repository](https://github.com/Google-Health/medsiglip)
- [MedGemma documentation](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [MedGemma model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [MedGemma repository](https://github.com/google-health/medgemma)
- [RAD-DINO model card](https://huggingface.co/microsoft/rad-dino)
- [CT-FM](https://github.com/project-lighter/CT-FM)
- [FlexiCT](https://github.com/ricklisz/FlexiCT)
- [Merlin](https://github.com/StanfordMIMI/Merlin)
- [Triad](https://github.com/wangshansong1/Triad)
- [NV-Segment-CTMR](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR)
- [MedSAM2](https://medsam2.github.io/)
- [M3D](https://github.com/BAAI-DCAI/M3D)

## Pathology references

- [H-Optimus](https://www.bioptimus.com/h-optimus)
- [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)
- [TITAN](https://github.com/mahmoodlab/TITAN)
- [CONCH](https://github.com/mahmoodlab/CONCH)

## Evaluation references

- [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/)
- [MONAI metrics](https://docs.monai.io/en/stable/metrics.html)
- [RadGraph](https://github.com/Stanford-AIMI/radgraph)

## Source-use checklist

- [ ] Read the upstream model card, repository README, license, and inference code before writing an adapter.
- [ ] Pin a tested commit or model revision rather than relying on a branch head.
- [ ] Copy preprocessing behavior from reviewed source code/configuration, not memory.
- [ ] Record upstream custom operators and accelerator restrictions.
- [ ] Add the exact source URL and revision to the adapter's registry record.
- [ ] Add a regression fixture that detects upstream output/preprocess drift.
- [ ] Re-review links, licenses, and supported runtime versions before each release.

