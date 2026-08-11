#!/usr/bin/env python
"""Evaluate a MedGemma 1.5 LoRA adapter on MedReason data.

The labeled holdout is drawn only from the released training pool. The
participant-facing validation split is inference-only because it contains no
answers. This script loads an existing adapter and never updates weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

from scripts.medreason_data import (
    load_and_select_examples,
    parse_mcq_label,
    save_json,
    sha256_file,
)
from scripts.train_medgemma15 import DEFAULT_MODEL, MEDGEMMA_TERMS_URL
from scripts.train_vlm_unsloth import evaluate_generation, generate_one, set_model_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, default=Path("../Data/Train/medreason_train_selection.json"))
    parser.add_argument("--train-images", type=Path, default=Path("../Data/Train/imgs"))
    parser.add_argument(
        "--participant-json",
        type=Path,
        default=Path("../Data/Valid/medreason_validation_participant_facing.json"),
    )
    parser.add_argument("--participant-images", type=Path, default=Path("../Data/Valid/imgs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/medreason/medgemma15_adapter_eval"),
    )
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--dev-limit", type=int, default=32)
    parser.add_argument("--participant-limit", type=int, default=64)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--pan-and-scan", action="store_true")
    return parser.parse_args()


def load_adapter(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("MedGemma adapter evaluation requires CUDA")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.adapter, padding_side="left")
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
    }
    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base = Gemma3ForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    set_model_cache(model, True)
    model.eval()
    return model, processor, device


def main() -> int:
    args = parse_args()
    started = time.time()
    all_train, selected, _, dev_examples, participant_examples = load_and_select_examples(
        train_json=args.train_json,
        train_images=args.train_images,
        participant_json=args.participant_json,
        participant_images=args.participant_images,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        participant_limit=args.participant_limit,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    model, processor, device = load_adapter(args)
    dev_metrics = evaluate_generation(model, processor, dev_examples, args.max_new_tokens)

    participant_predictions: list[dict[str, str]] = []
    for index, example in enumerate(participant_examples, 1):
        generated = generate_one(model, processor, example, args.max_new_tokens)
        participant_predictions.append(
            {
                "case_id": example.case_id,
                "task_type": example.task_type,
                "answer": parse_mcq_label(generated) if example.task_type == "mcq" else generated,
                "generated": generated,
            }
        )
        if index % 16 == 0:
            print("participant_inference", index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "local_dev_predictions.json", dev_metrics["outputs"])
    save_json(args.output_dir / "participant_predictions.json", participant_predictions)
    save_json(
        args.output_dir / "results.json",
        {
            "name": "MedReason MedGemma 1.5 predictions",
            "type": "Medical visual reasoning",
            "version": {"major": 1, "minor": 0},
            "answers": [
                {
                    "case_id": row["case_id"],
                    "task_type": row["task_type"],
                    "answer": row["answer"],
                    "reasoning_trace": row["generated"] if row["task_type"] == "open-ended" else row["generated"],
                    "confidence": 1.0,
                }
                for row in participant_predictions
            ],
        },
    )
    summary = {
        "run_kind": "medreason_medgemma15_adapter_evaluation",
        "model": args.model,
        "adapter": str(args.adapter),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "medgemma_terms": MEDGEMMA_TERMS_URL,
        "train_source_cases": len(all_train),
        "selected_cases": len(selected),
        "local_dev_cases": len(dev_examples),
        "participant_inference_cases": len(participant_examples),
        "local_dev_generation": {key: value for key, value in dev_metrics.items() if key != "outputs"},
        "participant_labels_available": False,
        "four_bit": not args.no_4bit,
        "attn_implementation": args.attn_implementation,
        "pan_and_scan": args.pan_and_scan,
        "gpu": torch.cuda.get_device_name(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json(args.output_dir / "summary.json", summary)
    print("summary", json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
