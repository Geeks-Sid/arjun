"""Pipeline composition: deterministic stage, stochastic stage, cache boundary.

A :class:`TransformPipeline` is an explicit two-stage composition:

1. **Deterministic stage** — decode/canonicalization/normalization. Its output
   is a pure function of (payload, config), so the pipeline's
   :meth:`deterministic_config_hash` is the cache key component: the cache
   boundary sits *after* this stage. Stochastic transforms never contribute
   to it — no stochastic transform can contaminate a cache key.
2. **Stochastic stage** — augmentation, drawing only from the seeded
   ``TransformContext``. Runs post-cache, per worker/epoch/sample.

After both stages the final tensor is validated against the selected
:class:`~medfm.data.transforms.specs.PreprocessSpec` (when one is attached),
guaranteeing adapters receive exactly their declared tensor format.

Execution is host-only by design: decode, canonicalization, and unsupported
medical transforms stay on the CPU; fixed tensors transfer to the accelerator
only after collation. Accelerator execution of transforms is opt-in and
gated behind parity tests before it may be enabled (see
``accelerator_training_strategy.md``); this module implements no device
execution path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from medfm.core.serialization import config_hash
from medfm.data.errors import TransformError
from medfm.data.transforms.base import Transform, TransformContext, TransformData
from medfm.data.transforms.specs import PreprocessSpec


class TransformPipeline:
    """Two-stage transform composition with an explicit cache boundary."""

    def __init__(
        self,
        deterministic: Sequence[Transform],
        stochastic: Sequence[Transform] = (),
        spec: PreprocessSpec | None = None,
        *,
        name: str = "pipeline",
    ) -> None:
        for transform in deterministic:
            if transform.stage != "deterministic":
                raise TransformError(
                    f"transform {transform.name!r} is declared stochastic but was placed in the "
                    "deterministic (cacheable) stage"
                )
        for transform in stochastic:
            if transform.stage != "stochastic":
                raise TransformError(
                    f"transform {transform.name!r} is declared deterministic but was placed in the stochastic stage; "
                    "deterministic transforms belong before the cache boundary"
                )
        self._deterministic = tuple(deterministic)
        self._stochastic = tuple(stochastic)
        self.spec = spec
        self.name = name

    @property
    def deterministic_transforms(self) -> tuple[Transform, ...]:
        return self._deterministic

    @property
    def stochastic_transforms(self) -> tuple[Transform, ...]:
        return self._stochastic

    def run_deterministic(self, data: TransformData) -> TransformData:
        """Apply the cacheable deterministic stage only."""
        result = data
        for transform in self._deterministic:
            result = transform(result, None)
        return result

    def run_stochastic(self, data: TransformData, ctx: TransformContext) -> TransformData:
        """Apply the augmentation stage with a seeded per-sample context."""
        result = data
        for transform in self._stochastic:
            result = transform(result, ctx)
        return result

    def __call__(self, data: TransformData, ctx: TransformContext | None = None) -> TransformData:
        """Run both stages and validate the final tensor against the spec."""
        result = self.run_deterministic(data)
        if self._stochastic:
            if ctx is None:
                raise TransformError(
                    f"pipeline {self.name!r} has stochastic transforms and requires a seeded TransformContext"
                )
            result = self.run_stochastic(result, ctx)
        if self.spec is not None:
            self.spec.validate(result.image)
        return result

    def config_dict(self) -> dict[str, Any]:
        """Full JSON-able configuration (both stages + spec)."""
        return {
            "name": self.name,
            "deterministic": [{"name": t.name, "config": t.config_dict()} for t in self._deterministic],
            "stochastic": [{"name": t.name, "config": t.config_dict()} for t in self._stochastic],
            "spec": self.spec.config_dict() if self.spec is not None else None,
        }

    def config_hash(self) -> str:
        """SHA-256 over the full pipeline configuration."""
        return config_hash(self.config_dict())

    def deterministic_config_dict(self) -> dict[str, Any]:
        """Configuration of the cacheable stage only — stochastic transforms
        and augmentation draws are excluded by construction."""
        return {
            "name": self.name,
            "deterministic": [{"name": t.name, "config": t.config_dict()} for t in self._deterministic],
            "spec": self.spec.config_dict() if self.spec is not None else None,
        }

    def deterministic_config_hash(self) -> str:
        """Cache-key component: identity of everything before the cache boundary."""
        return config_hash(self.deterministic_config_dict())
