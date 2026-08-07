from __future__ import annotations

from pathlib import Path

import pytest

from medfm.training.config import RunConfig

RECIPE_ROOT = Path(__file__).parents[2] / "configs" / "recipes" / "3d"


@pytest.fixture
def phase14_config(tmp_path: Path):
    def load(name: str) -> RunConfig:
        config = RunConfig.load(RECIPE_ROOT / name)
        values = config.to_dict()
        values["output_dir"] = str(tmp_path / Path(name).stem)
        values["max_steps"] = 1
        values["epochs"] = 1
        # A CPU contract test must not try to initialize CUDA/XLA simply
        # because the accepted recipe is also published for that backend.
        values["accelerator"] = {**values["accelerator"], "backend": "cpu", "distribution": "single"}
        return RunConfig.from_dict(values)

    return load
