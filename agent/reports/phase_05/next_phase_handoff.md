# Phase 05 → Phase 06+ Handoff

## Registry contract (frozen: REGISTRY_SCHEMA_VERSION = 1)

Adapter phases register `medfm.registry.schema.ModelSpec` records and
`medfm.registry.plugins.ModelPlugin` implementations. The registry internals
do not change for phases 06-08.

### Adapter registration instructions

1. Build a `ModelSpec` (see `medfm/registry/catalog.py` for field-by-field
   examples):
   - `revision`: exact 40+ hex commit SHA (mandatory for READY).
   - `license`: `LicenseSpec` with `APPROVED` status only after the phase-00
     record is `approved_*`; `approved_use_cases` mandatory. Gated models:
     `gated=True, acceptance_required=True, terms_url=...`.
   - `capabilities`: modalities/tasks as tuples; `output_types` must agree
     with tasks (validated: classification/retrieval/contrastive need
     POOLED_EMBEDDINGS; segmentation needs FEATURE_MAPS or SPATIAL_TOKENS;
     generative tasks need NATIVE_TEXT). Set `spatial_tokens_status`
     (NATIVE/HOOKED/UNAVAILABLE), `cuda_only_extensions`,
     `pure_pytorch_fallback`, `custom_operators`. PEFT: list
     `known_target_modules` or set `unknown_family_confirmation_required`.
   - `memory`: per-`LoadingMode` `MemoryEstimate` (host/device bytes,
     WeightFormat, Topology, cpu_offload, frozen_cache, uncertainty_note).
     `compile_risk_note` is separate from memory estimates. NF4/INT8 loading
     modes are CUDA-only — `validate_backend` rejects them on xla_tpu.
   - `preprocess`: the adapter's `PreprocessSpec` from Phase 04
     (normalization mandatory: one mean/std per channel).
   - `backend_support`: ALL five keys required (cpu, cuda_single,
     cuda_distributed, tpu_single_host, tpu_multi_host), each starting
     `BackendSupport()` (UNTESTED). Never claim SUPPORTED_* without smoke
     evidence — the schema rejects it.
2. `ModelRegistry.register(spec)` — duplicate IDs and conflicting/whitespace
   aliases raise. Aliases are optional.
3. Register a plugin so smoke/validation can construct the model:

   ```python
   from medfm.registry.plugins import ModelPlugin, register_plugin


   class MyAdapterPlugin(ModelPlugin):
       def build(self, spec): ...  # smallest meaningful nn.Module
       def tiny_input(self, spec): ...  # {"image": torch.zeros(1, C, *shape)}


   register_plugin("my-model-id", MyAdapterPlugin())
   ```

   `medfm/registry/plugins.py::DummyPlugin` is the complete reference
   example (tiny 2D CNN + named `image` input); the dummy model
   `dummy-tiny-2d` in `tests/phase_05/test_smoke.py::register_dummy` shows a
   full ModelSpec.
4. Smoke: `medfm models smoke <id> --backend cpu --artifact-dir <dir>` (or
   `medfm accelerator validate-model <id> --backend <key>`) runs the tiny
   backend-specific forward, writes a run artifact (exact model ID +
   revision, always), and promotes ONLY that backend's status via
   `record_backend_result` (revision must match the registered SHA).

## Accelerator capability status vocabulary (frozen)

`BackendStatus`: UNTESTED, CPU_CONTRACT_ONLY, SUPPORTED_SINGLE_DEVICE,
SUPPORTED_REPLICATED, SUPPORTED_SHARDED, BLOCKED_CUSTOM_OP, BLOCKED_MEMORY,
BLOCKED_UPSTREAM, NOT_APPLICABLE. Backend keys: cpu, cuda_single,
cuda_distributed, tpu_single_host, tpu_multi_host.

Evidence requirements:
- SUPPORTED_* requires `smoke_revision` (== registered SHA) + `smoke_date`
  recorded by `ModelRegistry.record_backend_result` after a real smoke.
- Source inspection never promotes past UNTESTED. A CUDA success never
  mutates TPU status (per-backend atomic update; regression-tested).
- BLOCKED_* requires a structured `blocked_reason`.
- bitsandbytes NF4/INT8 are rejected on xla_tpu before allocation; BF16 LoRA
  is the TPU path (ADR 0009). CUDA-only extensions without a tested
  pure-PyTorch/SDPA fallback are TPU-ineligible.

## License checklist (per model, before READY)

- [ ] Phase-00 record reviewed: terms read at the pinned revision; status
      flipped to `approved_research`/`approved_commercial` in
      `model_registry/licenses.yaml` (approved_commercial requires
      commercial_use == "permitted").
- [ ] Gated repo: terms accepted as a named individual on the provider site,
      then `medfm models accept-terms <id> --by <name>` (records outside
      source control; downloads fail closed until then).
- [ ] Exact commit SHA pinned in the catalog entry (replaces the
      `unresolved-pending-license-acceptance` placeholder).
- [ ] Expected file set + sha256 manifest recorded for
      `verify_file_hashes` (computed at pin time).
- [ ] `approved_use_cases` / `prohibited_use_cases` verified in
      `medfm models show <id>` output.

## Model smoke checklist (per backend, before SUPPORTED_*)

- [ ] Adapter plugin registered and builds at the pinned revision.
- [ ] `medfm accelerator validate-model <id> --backend <key>` passes with an
      artifact written (artifact carries exact model ID + revision).
- [ ] For TPU eligibility: no CUDA-only custom ops, or a tested
      pure-PyTorch/SDPA path; NF4 not used (BF16 LoRA instead).
- [ ] Evidence recorded: `record_backend_result` with matching revision;
      check `medfm models accelerator-report`.
- [ ] `measured_peak_bytes` captured from the smoke/run to refine the
      estimate (Phase 12).

## Blocked v1 checkpoints and owners (at phase 05 close)

All 17 roster models are BLOCKED; owner for every review is the Project
Maintainer (Siddhesh), review due 2026-11-02.

- License `pending_review` (5): rad-dino (MIT, confirm weights license at
  pinned revision), h-optimus-0 (Apache-2.0, gated acceptance needed), conch
  (non-commercial research license; RESEARCH class), titan (non-commercial;
  RESEARCH class), gemma-generic + qwen-generic (verify exact checkpoint
  license; qwen-generic is DEPLOYMENT-class pending review).
- License `blocked_unresolved` (12): medsiglip, medgemma-1.5-4b (HAI-DEF
  terms, named-individual acceptance), ct-fm, flexict-3d, merlin, m3d-lamed,
  triad, nv-segment-ctmr, brainiac (deferred, not v1), medsam2,
  gigapath-flash (gated, custom terms) — repository/weights/license terms
  unresolved; see `model_registry/licenses.yaml` notes per model.

## Weight operations (phases 06+)

- Download: `medfm models download <id> --cache-dir <dir>` — explicit
  network, pinned revision, safetensors-first, `*.py` excluded unless the
  spec carries a reviewed `trust_remote_code_allowed=True`, tokens never
  logged, partial downloads rejected.
- Local resolution without network: `medfm.registry.resolve_local_path`.
- Validation: `medfm models validate <id> --local-weights <dir>` (file set /
  partials), `medfm.registry.verify_file_hashes(dir, expected)` for exact
  sha256 enforcement.
