# Phase 05 Summary: Model Registry, Capabilities, and Weight Management

## Outcome

The model registry is implemented and gated: versioned capability schemas
(`ModelSpec`, `LicenseSpec`, `MemoryProfile`, `PeftCapability`,
`BackendSupport`), fail-closed license and governance gates, explicit and
integrity-checked weight acquisition with externally stored gated-access
acceptance, per-backend capability discovery with smoke evidence, a full
`medfm models` CLI plus `medfm accelerator validate-model`, and the complete
v1 roster (17 models) registered as BLOCKED with structured license reasons
(no v1 license is approved yet, so the deployment-ready catalog is correctly
empty). 50 phase-local tests pass on CPU; the full repository suite passes
(593 passed, 10 protected-hardware skips). `ruff check`, `ruff format
--check`, and strict `mypy` are clean.

## What was built

- `medfm/registry/schema.py` — frozen registry schema v1
  (`REGISTRY_SCHEMA_VERSION = 1`). `ModelSpec` enforces pinned commit-SHA
  revisions for READY models, complete license records, per-backend support
  keys (cpu, cuda_single, cuda_distributed, tpu_single_host, tpu_multi_host —
  the phase-00 vocabulary), safe aliases, deprecation/replacement metadata,
  and preprocess completeness (normalization present, one mean/std per
  channel). `ModelCapability` validates that declared outputs agree with
  declared tasks (classification/retrieval/contrastive → POOLED_EMBEDDINGS,
  segmentation → FEATURE_MAPS or SPATIAL_TOKENS, generative → NATIVE_TEXT)
  and rejects contradictory flags (cuda_only_extensions vs
  pure_pytorch_fallback). `BackendSupport` makes SUPPORTED_* statuses invalid
  without recorded smoke revision+date; BLOCKED_* statuses require reasons.
  WeightFormat (FP32/BF16/FP16/INT8/NF4), Topology
  (SINGLE_DEVICE/REPLICATED/SPMD_FSDP), cpu_offload/frozen_cache flags, and
  measured_peak_bytes cover the loading-mode/memory matrix; compile risk is
  recorded separately from memory estimates.
- `medfm/registry/core.py` — `ModelRegistry`: duplicate-ID and unsafe-alias
  rejection, queries by modality/task/loading mode/license class/backend
  (backend filter matches only evidence-backed SUPPORTED_* statuses),
  `record_backend_result` for atomic per-backend evidence updates (a CUDA
  success never mutates TPU status; revisions must match the registered
  SHA), `validate_backend` for pre-allocation rejection of unsupported
  model/mode/backend combinations (bitsandbytes INT8/NF4 rejected on
  xla_tpu; CUDA-only extensions without a tested pure-PyTorch path rejected
  on TPU), and `accelerator_report`.
- `medfm/registry/weights.py` — explicit `download_weights` (pinned
  revision, safetensors-first allowlist, pickle formats only with
  `--allow-unsafe`, `*.py` excluded unless `trust_remote_code_allowed`,
  atomic/resumable via huggingface_hub, tokens never logged, partial
  downloads rejected), `verify_file_hashes` (exact file set + sha256),
  `find_partial_downloads` / `verify_weight_integrity`, `inspect_weights`,
  and no-network `resolve_local_path`.
- `medfm/registry/acceptance.py` — gated-access acceptance recorded outside
  source control (`~/.cache/medfm/gated_access.json` or
  `MEDFM_GATED_ACCESS_FILE`), keyed to the exact repository; downloads of
  gated models fail closed until acceptance is recorded explicitly. Nothing
  auto-accepts terms; a successful download never implies acceptance.
- `medfm/registry/catalog.py` — the real v1 roster loaded from
  `model_registry/licenses.yaml` + `v1_scope.yaml`: license records mapped
  fail-closed (unresolved/pending → BLOCKED with owner and review date;
  non-commercial/conditional → RESEARCH class so they never enter the
  deployment catalog), per-backend statuses from the scope file, declared
  output capabilities per family, conservative pre-adapter memory estimates
  with uncertainty notes.
- `medfm/registry/plugins.py` + `medfm/registry/smoke.py` — the adapter
  hook for phases 06-08 (`ModelPlugin` protocol: `build` + `tiny_input`),
  a reference `DummyPlugin` (tiny 2D CNN), and `run_smoke`: tiny
  backend-specific forward pass on CPU/CUDA/TPU (CUDA/TPU guarded by
  availability), writing a run artifact that always carries the exact model
  ID and pinned revision, and recording per-backend smoke evidence on
  success. Models without adapters raise a structured `NoAdapterError`.
- `medfm/cli/models.py` — `medfm models list` (filters incl. `--backend`,
  blocked models shown with reasons by default, `--ready-only` view),
  `show` (incl. approved/prohibited use cases and per-backend status),
  `validate` (metadata; `--local-weights` adds local weight validation;
  neither uses the network), `download` (explicit network),
  `accept-terms`, `estimate-memory`, `smoke`, `inspect-modules`, and
  `accelerator-report`; JSON output throughout.
- `medfm/cli/accelerator.py` — `medfm accelerator validate-model
  <id> --backend <key>`: pre-allocation policy check followed by a tiny
  backend-specific smoke against the exact registered revision.
- `medfm/tools/validate_phase.py` — phase 05 gate: required files plus a
  v1-catalog consistency check (every roster model registered READY or
  BLOCKED with reason, per-backend keys complete).

## Test coverage (50 tests)

License gates (incomplete/unknown licenses fail closed; approved licenses
require use-case declarations; gated licenses require acceptance fields),
revision pinning (READY requires SHA; BLOCKED placeholders allowed),
capability contradictions and task/output agreement, missing normalization,
duplicate IDs and unsafe aliases, research/deployment catalog separation,
backend evidence rules (SUPPORTED requires smoke revision+date; CUDA success
leaves TPU UNTESTED; revision mismatch rejected), pre-allocation rejection
(NF4 on TPU; CUDA-only extensions on TPU; unknown backends/modes), weight
integrity (hash mismatch, missing/extra files, partial-download detection),
gated-access blocking until explicit external acceptance, no-network
metadata commands (snapshot_download hard-disabled), dummy-plugin smoke with
run-artifact model identity, blocked models never smoke, the full v1 catalog
registration, and CLI/accelerator-validate behavior.

## Gate status

- `pytest tests/phase_05 -q` → 50 passed.
- `pytest tests/ -q` → 593 passed, 10 protected-hardware skips.
- `ruff check` / `ruff format --check` / `mypy` (strict) → clean.
- Smoke: `python -m medfm.cli.models list --format json` → 17 roster models.
- `python -m medfm.tools.validate_phase --phase 05` → passed.
- REGISTRY_SCHEMA_VERSION frozen at 1 for adapter phases 06-08.
