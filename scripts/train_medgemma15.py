#!/usr/bin/env python
"""Unsloth MedGemma 1.5 VLM fine-tuning entrypoint for MedReason 2D images.

MedGemma's gated checkpoint is loaded through Unsloth's ``FastVisionModel``.
The former hand-written CUDA/PEFT optimizer loop is intentionally not kept:
all production 2D VLM SFT now goes through the shared Unsloth/TRL runner.
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

DEFAULT_MODEL = "google/medgemma-1.5-4b-it"
DEFAULT_OUTPUT_DIR = Path("artifacts/runs/medreason/unsloth_medgemma15_adapter")
MEDGEMMA_TERMS_URL = "https://developers.google.com/health-ai-developer-foundations/terms"


def parse_args():
    return build_parser(
        description=__doc__ or "Unsloth MedGemma 1.5 fine-tuning",
        default_model=DEFAULT_MODEL,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        model_family="medgemma-1.5",
    ).parse_args()


def main() -> int:
    return run_cli(parse_args(), model_family="medgemma-1.5", terms_url=MEDGEMMA_TERMS_URL)


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
    "MEDGEMMA_TERMS_URL",
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
