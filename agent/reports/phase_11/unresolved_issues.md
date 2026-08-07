# Unresolved or environment-blocked checks

- TPU/XLA execution was not run because `torch_xla` is not installed on the workstation. CPU/CUDA tensor paths, static-shape interfaces, and backend-neutral reduction code were tested without claiming TPU parity.
- A real DDP/FSDP/XLA process-group run was not part of the available acceptance environment. `reduce_mean_by_count` is covered with uneven true counts and exposes a reducer callback boundary for the Phase 12 trainer.
- Bare `python -m pytest -q` is not a valid repository-wide signal in this checkout: collection fails before tests with a missing `safetensors` dependency in Phase 03 and duplicate test module basenames between older phase directories. The isolated Phase 11 acceptance command passes.
- Advanced uncertainty/GradNorm weighting implementations remain extension points by design; Phase 11 provides the protocol and fixed/scheduled implementations, not a training-loop recipe.
