#!/usr/bin/env python
"""Reload a local Qwen2.5-VL adapter and evaluate it offline.

The labeled holdout is drawn only from the training pool. Participant-facing
validation is inference-only because its released records contain no answers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from scripts.train_medreason_qwen import (
    DEFAULT_MODEL,
    DEFAULT_PARTICIPANT_IMAGES,
    DEFAULT_PARTICIPANT_JSON,
    DEFAULT_TRAIN_IMAGES,
    DEFAULT_TRAIN_JSON,
    evaluate_generation,
    generate_one,
    load_examples,
    parse_mcq_label,
    select_examples,
    split_train_dev,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--participant-json", type=Path, default=DEFAULT_PARTICIPANT_JSON)
    parser.add_argument("--participant-images", type=Path, default=DEFAULT_PARTICIPANT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/medreason/local_qwen_pilot_eval"))
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--dev-limit", type=int, default=32)
    parser.add_argument("--participant-limit", type=int, default=64)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_adapter(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("adapter evaluation requires CUDA")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(
        args.adapter,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    if not args.no_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    return model, processor, device


def main() -> int:
    args = parse_args()
    started = time.time()
    all_train = load_examples(args.train_json, args.train_images, labeled=True)
    selected = select_examples(all_train, args.train_limit)
    _, dev_examples = split_train_dev(selected, args.dev_fraction, args.seed)
    dev_examples = select_examples(dev_examples, args.dev_limit)
    participant_examples = select_examples(
        load_examples(args.participant_json, args.participant_images, labeled=False),
        args.participant_limit,
    )
    model, processor, device = load_adapter(args)
    dev_metrics = evaluate_generation(model, processor, dev_examples, device, args.max_new_tokens)
    participant_predictions: list[dict[str, str]] = []
    for index, example in enumerate(participant_examples, 1):
        generated = generate_one(model, processor, example, device, args.max_new_tokens)
        participant_predictions.append(
            {
                "case_id": example.case_id,
                "task_type": example.task_type,
                "answer": parse_mcq_label(generated) or "" if example.task_type == "mcq" else generated,
                "generated": generated,
            }
        )
        if index % 16 == 0:
            print("participant_inference", index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "local_dev_predictions.json", dev_metrics["outputs"])
    save_json(args.output_dir / "participant_predictions.json", participant_predictions)
    summary = {
        "run_kind": "local_diagnostic_qwen2.5_vl_3b_adapter_evaluation",
        "adapter": str(args.adapter),
        "model": str(args.model),
        "selected_cases": len(selected),
        "local_dev_cases": len(dev_examples),
        "participant_inference_cases": len(participant_examples),
        "local_dev_generation": {key: value for key, value in dev_metrics.items() if key != "outputs"},
        "participant_mcq_nonempty": sum(
            bool(row["answer"]) for row in participant_predictions if row["task_type"] == "mcq"
        ),
        "participant_labels_available": False,
        "four_bit": not args.no_4bit,
        "gpu": torch.cuda.get_device_name(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json(args.output_dir / "summary.json", summary)
    print("summary", json.dumps(summary, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
