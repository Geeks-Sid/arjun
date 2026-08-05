# Phase 03 Unresolved Issues

1. **cuCIM slide backend not exercised on this host.** `CuCIMSlideReader` is
   capability-gated and its constructor correctly raises
   `UnsupportedFormatError` when cuCIM is absent (tested). The actual cuCIM
   decode path needs an NVIDIA CUDA host with the optional `cucim` extra
   installed; it is untested here by design (CPU/TPU hosts never require it).

2. **Vendor WSI formats untested against real scanner files.** The slide
   contract is validated on a synthetic OME-TIFF pyramid read by both
   OpenSlide and TiffSlide. Real vendor formats (SVS/NDPI/MRXS) and their
   embedded metadata were deliberately not ingested (no patient data). OpenSlide
   reports the synthetic pyramid as a single level (generic-TIFF behavior),
   while TiffSlide exposes all levels.

3. **Distributed sampling verified in-process, not across processes.** The
   group-aware sampler's disjoint-coverage, group-integrity, and padding
   guarantees are tested by materializing every rank's shard in one process.
   A true multi-process `torch.distributed` run needs the protected
   distributed runner (`MEDFM_RUN_DISTRIBUTED_TESTS=1`), which is not
   available on this workstation.

4. **Compressed DICOM transfer syntaxes not exercised.** Synthetic series use
   uncompressed Explicit VR Little Endian. JPEG/RLE-compressed pixel data
   depends on pydicom's pixel handlers and is not covered by synthetic
   fixtures; unsupported decoders already fail with actionable
   `UnsupportedFormatError`s.

None of these block Phase 04. Items 1–3 are tracked for the GPU/TPU and
distributed acceptance runs in later phases; item 4 will be revisited if real
compressed DICOM ingestion becomes a requirement.
