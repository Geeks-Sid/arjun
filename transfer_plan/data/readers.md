# transfer_plan/data/readers.md

Covers: `data/readers/{base,dicom,radiology,pathology}.py`.
Wave: 2 — **keep by design**. These already wrap the right libraries; no re-implementation.

## Transfer checklist

- [x] `radiology.py` (`NiftiReader`, `MHAReader`, `NumpyVolumeReader`, `PngJpegReader`) →
      **keep** — already delegate to nibabel / SimpleITK / Pillow; per-file dtype mapping
      (`_NUMPY_TO_TORCH`, `_torch_from_numpy`) is the medfm contract (canonical dtype names).
- [x] `dicom.py` (`DICOMSeriesReader`, `discover_dicom_series`) → **keep** — pydicom already;
      series grouping, orientation/spacing tolerances, slice-gap validation are contract logic
      MONAI's `LoadImage`/`dcm_*` don't reproduce.
- [x] `pathology.py` (`OpenSlideReader`, `TiffSlideReader`, `CuCIMSlideReader`,
      `PreExtractedTileReader`, `EmbeddingStoreReader`) → **keep** — already wrap
      openslide/tiffslide/cucim; `convert_level_coords`/`validate_level_coords` are thin math
      on `PyramidLevel`; tile-store readers read safetensors/HDF5 (library-backed).
- [x] `base.py` (`Reader`, `hash_identifier`, privacy-safe metada) → **keep** — contract.


## Tests
`tests/phase_03/test_dicom_reader.py`, `test_radiology_readers.py`, `test_pathology_readers.py`.

## Result

All four reader items verified as **keep**: readers already wrap nibabel, SimpleITK, Pillow,
pydicom, OpenSlide/TiffSlide/cuCIM, safetensors, and HDF5 while preserving medfm geometry, dtype,
privacy, and tile-coordinate contracts. No transfer or parity drift measured. Source and test files
were unchanged; only this checklist was updated.

Validation (shared phase run): `uv run --frozen pytest tests/phase_02 tests/phase_03 tests/phase_04
tests/phase_05 tests/phase_06 tests/phase_07 tests/phase_09` — **PASS** (622 passed, 4 skipped,
1 warning). Scoped `ruff check` — **PASS**; `ruff format --check` — **PASS** (43 files);
scoped `mypy` — **PASS**.
