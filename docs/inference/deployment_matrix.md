# Phase 17 deployment matrix

The bundle manifest is the source of truth for a particular export. A runtime
must reject a backend whose status is `untested`, `blocked`, or `prohibited`.
`tested` is evidence for the declared model/task bundle, not a framework-wide
claim.

| Backend | Canonical loading | Inference policy | Required evidence |
| --- | --- | --- | --- |
| CPU | CPU safetensors, FP32 | Always available for bundles declaring `tested` | Bundle load, output parity, memory cap |
| CUDA | CPU safetensors then backend placement | FP32/BF16/FP16 only when bundle declares support; no hard-coded `.cuda()` path | Cold/warm latency, peak VRAM, output parity |
| TPU/XLA | CPU safetensors then XLA placement | Fixed predeclared token/image/volume/tile buckets; pad or reject out-of-bucket requests | Bucket warmup, compile count, host synchronization, steady-state latency |

## Prohibited combinations

- Resumable distributed checkpoints (`optimizer`, sharded `.distcp`, RNG, or
  scheduler state) are not deployment inputs. Convert explicitly to canonical
  adapter/bridge/head safetensors first.
- Quantized CUDA-only tensors and custom CUDA operators are not portable TPU
  artifacts. A bundle must declare the restriction rather than falling back.
- Merged base-weight exports are secondary convenience artifacts. They are not
  accepted as the canonical adapter source and require a documented conversion
  record.
- Unreviewed model Python, arbitrary import paths, and arbitrary request file
  paths are not part of the serving API.

## Warmup and capacity runbook

1. Validate `bundle.json`, metadata hashes, all file SHA-256 checksums, base
   revision, and backend status before allocating a model.
2. Load only the requested adapter. Keep CUDA and TPU worker pools/configuration
   separate even though request/response schemas are shared.
3. Warm every declared XLA bucket before readiness and latency measurement.
   Report compile warmup separately from CUDA cold/warm latency.
4. Enforce request, output-token, tile, volume, and memory caps; return a
   structured error without input payloads.
5. Persist required operational audit fields (model/adapter revision,
   preprocessing hash, prompt version, input hash, schema, runtime, memory,
   and error status). Raw images, reports, and raw DICOM UIDs remain in the
   access-controlled clinical audit store only.
