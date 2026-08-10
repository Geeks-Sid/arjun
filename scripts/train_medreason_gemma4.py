#!/usr/bin/env python
"""Local Gemma 4 E4B adapter pilot for the MedReason data layout.

This runner is separate from the protected MedReason route. It trains a small
adapter-only Gemma 4 multimodal model, evaluates a deterministic labeled
holdout, and performs label-free inference on participant-facing validation.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from scripts.train_medreason_qwen import (
    DEFAULT_PARTICIPANT_IMAGES,
    DEFAULT_PARTICIPANT_JSON,
    DEFAULT_TRAIN_IMAGES,
    DEFAULT_TRAIN_JSON,
    Example,
    ExampleDataset,
    build_prompt,
    build_target,
    evaluate_loss,
    load_examples,
    move_batch,
    optimizer_and_scheduler,
    parse_mcq_label,
    save_json,
    select_examples,
    sha256_file,
    split_train_dev,
)

DEFAULT_MODEL = Path("artifacts/models/local/gemma-4-E4B-it")


class Gemma4VisionLanguageCollator:
    def __init__(self, processor: Any, max_seq_length: int, max_soft_tokens: int) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.processor_kwargs = {"images_kwargs": {"max_soft_tokens": max_soft_tokens}}

    @staticmethod
    def _messages(example: Example, target: str | None = None) -> list[dict[str, Any]]:
        content: list[dict[str, str]] = [
            {"type": "image", "url": str(example.image_path)},
            {"type": "text", "text": build_prompt(example)},
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if target is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        return messages

    def _apply(self, example: Example, *, target: str | None, generation: bool) -> Any:
        return self.processor.apply_chat_template(
            self._messages(example, target),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=generation,
            return_assistant_tokens_mask=False,
            enable_thinking=False,
            processor_kwargs=self.processor_kwargs,
        )

    def __call__(self, examples: list[Example]) -> dict[str, Tensor]:
        if len(examples) != 1:
            raise ValueError("Gemma 4 pilot collator requires batch_size=1")
        example = examples[0]
        full = self._apply(example, target=build_target(example), generation=False)
        assistant_masks = full.pop("assistant_masks", None)
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        if input_ids.shape[1] > self.max_seq_length:
            raise ValueError(
                f"processed example exceeds max_seq_length={self.max_seq_length}; "
                "increase the bucket instead of truncating"
            )
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if assistant_masks is not None:
            mask = assistant_masks if isinstance(assistant_masks, Tensor) else torch.tensor(assistant_masks)
        else:
            mask = torch.zeros_like(labels)
        if mask.any():
            labels[mask == 0] = -100
        else:
            prompt = self._apply(example, target=None, generation=True)
            prompt_length = int(prompt["attention_mask"].sum().item())
            if prompt_length > input_ids.shape[1]:
                raise ValueError("assistant boundary is outside the processed sequence")
            labels[:, :prompt_length] = -100
        full["labels"] = labels
        return {str(key): value for key, value in full.items() if isinstance(value, Tensor)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--participant-json", type=Path, default=DEFAULT_PARTICIPANT_JSON)
    parser.add_argument("--participant-images", type=Path, default=DEFAULT_PARTICIPANT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/medreason/local_gemma4_e4b_pilot"))
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--dev-limit", type=int, default=32)
    parser.add_argument("--participant-limit", type=int, default=64)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-soft-tokens", type=int, default=70)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-participant-inference", action="store_true")
    return parser.parse_args()


def set_cache(model: Any, enabled: bool) -> None:
    for config in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = enabled
def gemma_target_modules(model: torch.nn.Module) -> list[str]:
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = {
        f"{name.rsplit('.', 2)[-2]}.linear"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.endswith(".linear")
        and name.rsplit(".", 2)[-2] in wanted
    }
    selected = sorted(found)
    if not selected:
        raise RuntimeError("could not discover supported Gemma 4 LoRA target modules")
    return selected


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, torch.device, list[str]]:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("the local pilot requires CUDA; use the protected training path elsewhere")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model)
    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForMultimodalLM.from_pretrained(args.model, **model_kwargs)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    set_cache(model, False)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modules = gemma_target_modules(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=modules,
        ),
    )
    model.print_trainable_parameters()
    return model, processor, device, modules


def decode_generation(processor: Any, output_ids: Tensor, input_ids: Tensor) -> str:
    generated = output_ids[0, input_ids.shape[1] :]
    try:
        parsed = processor.parse_response(generated, prefix=input_ids[0])
        if isinstance(parsed, dict):
            content = parsed.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_one(
    model: Any,
    processor: Any,
    example: Example,
    device: torch.device,
    max_new_tokens: int,
    max_soft_tokens: int = 70,
) -> str:
    messages = Gemma4VisionLanguageCollator._messages(example)
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
        processor_kwargs={"images_kwargs": {"max_soft_tokens": max_soft_tokens}},
    )
    batch = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}
    was_training = bool(model.training)
    previous_cache = getattr(getattr(model, "config", None), "use_cache", False)
    model.eval()
    set_cache(model, True)
    try:
        with torch.inference_mode():
            output_ids = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        set_cache(model, previous_cache)
        model.train(was_training)
    return decode_generation(processor, output_ids, batch["input_ids"])


def evaluate_generation(
    model: Any,
    processor: Any,
    examples: Iterable[Example],
    device: torch.device,
    max_new_tokens: int,
    max_soft_tokens: int = 70,
) -> dict[str, Any]:
    mcq_total = 0
    mcq_correct = 0
    open_total = 0
    outputs: list[dict[str, str]] = []
    for index, example in enumerate(examples, 1):
        raw = generate_one(model, processor, example, device, max_new_tokens, max_soft_tokens)
        if example.task_type == "mcq":
            predicted = parse_mcq_label(raw)
            mcq_total += 1
            mcq_correct += int(predicted == example.answer)
            outputs.append(
                {
                    "case_id": example.case_id,
                    "predicted_label": predicted or "",
                    "target_label": example.answer or "",
                    "generated": raw,
                }
            )
        else:
            open_total += 1
            outputs.append({"case_id": example.case_id, "generated": raw})
        if index % 16 == 0:
            print("generation_eval", index)
    return {
        "mcq_cases": mcq_total,
        "mcq_correct": mcq_correct,
        "mcq_accuracy": (mcq_correct / mcq_total) if mcq_total else None,
        "open_cases": open_total,
        "outputs": outputs,
    }


def main() -> int:
    args = parse_args()
    if args.train_limit == 0 or args.dev_limit == 0 or args.participant_limit == 0:
        raise ValueError("limits must be positive, or negative for all examples")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    all_train = load_examples(args.train_json, args.train_images, labeled=True)
    selected = select_examples(all_train, args.train_limit)
    train_examples, dev_examples = split_train_dev(selected, args.dev_fraction, args.seed)
    dev_examples = select_examples(dev_examples, args.dev_limit)
    participant_examples = load_examples(args.participant_json, args.participant_images, labeled=False)
    participant_examples = select_examples(participant_examples, args.participant_limit)
    print(
        "dataset",
        {
            "train_source_cases": len(all_train),
            "selected_cases": len(selected),
            "train_cases": len(train_examples),
            "local_dev_cases": len(dev_examples),
            "participant_inference_cases": len(participant_examples),
        },
    )

    model, processor, device, modules = load_model_and_processor(args)
    collator = Gemma4VisionLanguageCollator(processor, args.max_seq_length, args.max_soft_tokens)
    train_loader = DataLoader(ExampleDataset(train_examples), batch_size=1, shuffle=False, collate_fn=collator)
    dev_loader = DataLoader(ExampleDataset(dev_examples), batch_size=1, shuffle=False, collate_fn=collator)
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation)
    )
    optimizer, scheduler = optimizer_and_scheduler(model, args, total_steps)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    microsteps = 0
    losses: list[float] = []
    epoch = 0
    while epoch < args.epochs and step < total_steps:
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimizer step {step}")
            loss.backward()
            microsteps += 1
            losses.append(float(loss.detach().float().cpu()) * args.gradient_accumulation)
            if microsteps % args.gradient_accumulation == 0:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % 10 == 0:
                    peak = torch.cuda.max_memory_allocated() / (1024**3)
                    print("train_step", {"step": step, "loss": losses[-1], "peak_allocated_gib": round(peak, 3)})
                if args.eval_every > 0 and step % args.eval_every == 0:
                    dev_loss = evaluate_loss(model, dev_loader, device)
                    print("dev_loss", {"step": step, "loss": dev_loss})
                if step >= total_steps:
                    break
        epoch += 1
    if microsteps % args.gradient_accumulation:
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
    dev_loss = evaluate_loss(model, dev_loader, device)
    dev_metrics = evaluate_generation(model, processor, dev_examples, device, args.max_new_tokens, args.max_soft_tokens)

    set_cache(model, True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    participant_records: list[dict[str, str]] = []
    if not args.skip_participant_inference:
        for index, example in enumerate(participant_examples, 1):
            raw = generate_one(model, processor, example, device, args.max_new_tokens, args.max_soft_tokens)
            participant_records.append(
                {
                    "case_id": example.case_id,
                    "task_type": example.task_type,
                    "answer": parse_mcq_label(raw) if example.task_type == "mcq" else raw,
                }
            )
            if index % 16 == 0:
                print("participant_inference", index)
    save_json(args.output_dir / "participant_predictions.json", participant_records)
    model_class = (
        model.base_model.model.__class__.__name__
        if hasattr(model, "base_model")
        else model.__class__.__name__
    )
    summary = {
        "run_kind": "local_diagnostic_gemma4_e4b_adapter_pilot",
        "protected_medreason_route": False,
        "seed": args.seed,
        "model": str(args.model),
        "model_class": model_class,
        "train_json_sha256": sha256_file(args.train_json),
        "participant_json_sha256": sha256_file(args.participant_json),
        "train_source_cases": len(all_train),
        "selected_cases": len(selected),
        "train_cases": len(train_examples),
        "local_dev_cases": len(dev_examples),
        "participant_inference_cases": len(participant_examples),
        "optimizer_steps": step,
        "mean_train_loss": sum(losses) / len(losses) if losses else None,
        "local_dev_loss": dev_loss,
        "local_dev_generation": {key: value for key, value in dev_metrics.items() if key != "outputs"},
        "participant_labels_available": False,
        "lora_target_modules": modules,
        "four_bit": not args.no_4bit,
        "max_seq_length": args.max_seq_length,
        "max_soft_tokens": args.max_soft_tokens,
        "thinking_enabled": False,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": round(time.time() - start, 3),
    }
    save_json(args.output_dir / "local_dev_predictions.json", dev_metrics["outputs"])
    save_json(args.output_dir / "summary.json", summary)
    print("summary", {key: value for key, value in summary.items() if key != "lora_target_modules"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
