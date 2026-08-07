# transfer_plan/data/readers.md

Covers: `data/readers/{base,dicom,radiology,pathology}.py`.
Wave: 2 — **keep by design**. These already wrap the right libraries; no re-implementation.

## Transfer checklist

- [ ] `radiology.py` (`NiftiReader`, `MHAReader`, `NumpyVolumeReader`, `PngJpegReader`) →
      **keep** — already delegate to nibabel / SimpleITK / Pillow; per-file dtype mapping
      (`_NUMPY_TO_TORCH`, `_torch_from_numpy`) is the medfm contract (canonical dtype names).
- [ ] `dicom.py` (`DICOMSeriesReader`, `discover_dicom_series`) → **keep** — pydicom already;
      series grouping, orientation/spacing tolerances, slice-gap validation are contract logic
      MONAI's `LoadImage`/`dcm_*` don't reproduce.
- [ ] `pathology.py` (`OpenSlideReader`, `TiffSlideReader`, `CuCIMSlideReader`,
      `PreExtractedTileReader`, `EmbeddingStoreReader`) → **keep** — already wrap
      openslide/tiffslide/cucim; `convert_level_coords`/`validate_level_coords` are thin math
      on `PyramidLevel`; tile-store readers read safetensors/HDF5 (library-backed).
- [ ] `base.py` (`Reader`, `hash_identifier`, privacy-safe metada) → **keep** — contract.

## Tests
`tests/phase_03/test_dicom_reader.py`, `test_radiology_readers.py`, `test_pathology_readers.py`.
