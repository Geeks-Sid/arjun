"""Distributed loss reduction by true supervised counts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from medfm.core.errors import ShapeContractError
from medfm.core.task import LossOutput

ReduceFn = Callable[[torch.Tensor], torch.Tensor]

_torch_assert = cast(Callable[[Any, str], None], torch._assert)


def reduce_mean_by_count(
    local_mean: torch.Tensor,
    local_count: int | torch.Tensor,
    *,
    reduce_fn: ReduceFn | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce a local mean using summed true counts, not rank means.

    ``reduce_fn`` is intentionally a tiny backend boundary: DDP/XLA callers can
    pass an in-place/all-reduce adapter without importing either backend here.
    The returned count remains a tensor so the operation stays in the compiled
    graph and can represent uneven/padded batches correctly.
    """

    count = (
        local_count if isinstance(local_count, torch.Tensor) else torch.as_tensor(local_count, device=local_mean.device)
    )
    count = count.to(device=local_mean.device, dtype=torch.float32)
    if count.ndim != 0:
        raise ShapeContractError("local_count must be a non-negative scalar")
    _torch_assert(count >= 0, "local_count must be non-negative")
    stats = torch.stack([local_mean.float() * count, count])
    if reduce_fn is not None:
        stats = reduce_fn(stats)
    global_count = stats[1].clamp_min(1.0)
    return (stats[0] / global_count).to(dtype=local_mean.dtype), stats[1]


def reduce_loss_output(output: LossOutput, *, reduce_fn: ReduceFn | None = None) -> LossOutput:
    """Reduce total/components by the output's true sample count."""

    total, count = reduce_mean_by_count(output.total, output.sample_count, reduce_fn=reduce_fn)
    components: dict[str, torch.Tensor] = {}
    for name, value in output.components.items():
        components[name], _ = reduce_mean_by_count(value, output.sample_count, reduce_fn=reduce_fn)
    diagnostics: dict[str, Any] = dict(output.diagnostics)
    diagnostics["global_true_count"] = count.detach()
    diagnostics["reduction"] = "sum(loss * true_count) / sum(true_count)"
    return LossOutput(
        total=total,
        components=components,
        sample_count=output.sample_count,
        token_count=output.token_count,
        diagnostics=diagnostics,
    )


__all__ = ["ReduceFn", "reduce_mean_by_count", "reduce_loss_output"]
