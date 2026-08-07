# Phase 17 completion summary

Phase 17 delivers portable adapter-only deployment bundles, bounded inference
pipelines, reviewed medical-output exporters, and a privacy-safe serving layer.
The implementation is independent of the training run directory and routes
execution through the existing CPU/CUDA/TPU accelerator abstraction.

Implemented contracts include:

- versioned bundle manifests with pinned base-model revisions, runtime support,
  prohibited combinations, canonical CPU tensor artifacts, checksums, and
  secondary merged-artifact guards;
- bundle validation before model allocation, safe relative paths, resume-state
  rejection, and adapter-only loading through `BundleLoader`;
- classification, segmentation, retrieval, VLM, and WSI request pipelines with
  modality/task validation, preprocessing/postprocessing hooks, tensor/token/
  tile/volume limits, structured errors, hashes, and memory checks;
- bounded Gaussian sliding-window reconstruction and fixed TPU bucket padding /
  rejection policies with warmup support;
- original-shape/affine NIfTI export and reopen validation plus policy-gated
  highdicom SEG export with hashed source references;
- deterministic clinical-style VLM generation, explicit beam/research-sampling
  gates, prompt isolation, JSON-schema validation, output/stop/visual-token
  limits, uncertainty status, prompt-version recording, and bounded length
  buckets;
- adapter registration, requested-only loading, bounded LRU eviction, stale
  adapter-state reset, concurrency/backpressure, timeout handling, and common
  versioned request/response schemas;
- operational audit records containing hashes and bounded metadata only, with a
  separate role-gated clinical audit store and retention/deletion controls;
- published bundle, request, response, deployment-matrix, and license-catalog
  artifacts under `docs/inference/`.

The Phase 17 focused suite passes 15 tests. The smoke classifier command
produces a structured response. The acceptance validator passes after all
required report artifacts are present.
