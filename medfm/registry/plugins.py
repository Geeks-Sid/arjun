"""Model plugins: the adapter hook for phases 06-08, plus a dummy plugin.

A plugin knows how to build a tiny, smoke-testable instance of a registered
model and how to construct the tiny backend-specific input for it. Real
adapters register their plugin next to their ModelSpec; until then, only the
dummy plugin exists and real models smoke as "no adapter registered".

Adapter registration (contract for phases 06-08):

    from medfm.registry.plugins import register_plugin, ModelPlugin
    from medfm.registry.schema import ModelSpec

    class MyAdapterPlugin(ModelPlugin):
        def build(self, spec: ModelSpec) -> torch.nn.Module: ...
        def tiny_input(self, spec: ModelSpec) -> Mapping[str, torch.Tensor]: ...

    ModelRegistry.register(my_spec)
    register_plugin("my-model-id", MyAdapterPlugin())

See the DummyPlugin below for a complete minimal example.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from medfm.registry.schema import ModelSpec


@runtime_checkable
class ModelPlugin(Protocol):
    """Builds a smoke-testable model instance and its tiny input."""

    def build(self, spec: ModelSpec) -> Any:
        """Construct the smallest meaningful instance of the model."""
        ...

    def tiny_input(self, spec: ModelSpec) -> Mapping[str, Any]:
        """A single tiny example input (tensors on CPU; smoke moves them)."""
        ...


_PLUGINS: dict[str, ModelPlugin] = {}


def register_plugin(model_id: str, plugin: ModelPlugin) -> None:
    if model_id in _PLUGINS:
        raise ValueError(f"Duplicate plugin for model: {model_id}")
    if not isinstance(plugin, ModelPlugin):
        raise TypeError(f"plugin for {model_id} does not satisfy ModelPlugin protocol")
    _PLUGINS[model_id] = plugin


def get_plugin(model_id: str) -> ModelPlugin | None:
    return _PLUGINS.get(model_id)


def clear_plugins() -> None:
    """Clear all plugins (for testing); does not remove ModelSpecs."""
    _PLUGINS.clear()


class _DummyNet:
    """Minimal nn.Module accepting the named ``image`` input (adapter-style)."""

    def __new__(cls, channels: int) -> Any:
        import torch

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = torch.nn.Sequential(
                    torch.nn.Conv2d(channels, 4, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(4, 2),
                )

            def forward(self, image: Any) -> Any:
                return self.body(image)

        return Net()


class DummyPlugin:
    """Tiny 2D CNN plugin proving the registry smoke path end to end.

    Registered as ``dummy-tiny-2d``; used by Phase 05 tests and as the
    reference example for adapter phases.
    """

    def build(self, spec: ModelSpec) -> Any:
        channels = spec.preprocess.channels if spec.preprocess else 1
        return _DummyNet(channels)

    def tiny_input(self, spec: ModelSpec) -> Mapping[str, Any]:
        import torch

        shape = spec.preprocess.spatial_shape if spec.preprocess else (8, 8)
        channels = spec.preprocess.channels if spec.preprocess else 1
        return {"image": torch.zeros(1, channels, *shape)}
