#!/usr/bin/env python
"""Unsloth Qwen2.5-VL fine-tuning entrypoint for MedReason 2D images.

The previous hand-written PyTorch/PEFT loop has been removed.  This entrypoint
now uses ``FastVisionModel`` + ``UnslothVisionDataCollator`` + ``SFTTrainer``
through :mod:`scripts.train_vlm_unsloth`.
"""

from __future__ import annotations

from pathlib import Path

from scripts.medreason_data import (
    DEFAULT_PARTICIPANT_IMAGES,
    DEFAULT_PARTICIPANT_JSON,
    DEFAULT_TRAIN_IMAGES,
    DEFAULT_TRAIN_JSON,
    Example,
    build_messages,
    build_prompt,
    build_target,
    case_to_example,
    load_and_select_examples,
    load_examples,
    load_image,
    parse_mcq_label,
    save_json,
    select_examples,
    sha256_file,
    split_train_dev,
)
from scripts.train_vlm_unsloth import (
    UnslothVLMConfig,
    UnslothVLMError,
    build_parser,
    build_unsloth_trainer,
    evaluate_generation,
    generate_one,
    run_cli,
    set_model_cache,
    train_unsloth_vlm,
)

DEFAULT_MODEL = "unsloth/Qwen2.5-VL-3B-Instruct"
DEFAULT_OUTPUT_DIR = Path("artifacts/runs/medreason/unsloth_qwen25_vl_adapter")


def parse_args():
    return build_parser(
        description=__doc__ or "Unsloth Qwen2.5-VL fine-tuning",
        default_model=DEFAULT_MODEL,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        model_family="qwen2.5-vl",
        default_min_pixels=256 * 28 * 28,
        default_max_pixels=512 * 28 * 28,
    ).parse_args()


def main() -> int:
    return run_cli(parse_args(), model_family="qwen2.5-vl")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PARTICIPANT_IMAGES",
    "DEFAULT_PARTICIPANT_JSON",
    "DEFAULT_TRAIN_IMAGES",
    "DEFAULT_TRAIN_JSON",
    "Example",
    "UnslothVLMConfig",
    "UnslothVLMError",
    "build_messages",
    "build_parser",
    "build_prompt",
    "build_target",
    "build_unsloth_trainer",
    "case_to_example",
    "evaluate_generation",
    "generate_one",
    "load_and_select_examples",
    "load_examples",
    "load_image",
    "main",
    "parse_args",
    "parse_mcq_label",
    "run_cli",
    "save_json",
    "select_examples",
    "set_model_cache",
    "sha256_file",
    "split_train_dev",
    "train_unsloth_vlm",
]
