#!/usr/bin/env python
"""Local 24-GiB Qwen2.5-VL pilot for the MedReason data layout.

This runner is intentionally separate from the protected MedReason route in the
implementation plan. It trains a small, adapter-only local model, evaluates on
a deterministic holdout made from the labeled training pool, and performs
label-free inference on participant-facing validation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

LABELS = tuple("ABCDE")
DEFAULT_MODEL = Path("artifacts/models/local/qwen2.5-vl-3b-instruct")
DEFAULT_TRAIN_JSON = Path("../Data/Train/medreason_train_selection.json")
DEFAULT_TRAIN_IMAGES = Path("../Data/Train/imgs")
DEFAULT_PARTICIPANT_JSON = Path("../Data/Valid/medreason_validation_participant_facing.json")
DEFAULT_PARTICIPANT_IMAGES = Path("../Data/Valid/imgs")


@dataclass(frozen=True)
class Example:
    case_id: str
    task_type: str
    question: str
    options: tuple[tuple[str, str], ...]
    answer: str | None
    image_path: Path


class ExampleDataset(Dataset[Example]):
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


class VisionLanguageCollator:
    def __init__(self, processor: Any, max_seq_length: int) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length

    @staticmethod
    def _messages(example: Example, target: str | None = None) -> list[dict[str, Any]]:
        content: list[dict[str, str]] = [{"type": "image", "image": str(example.image_path)}]
        content.append({"type": "text", "text": build_prompt(example)})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if target is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        return messages

    def __call__(self, examples: list[Example]) -> dict[str, Tensor]:
        full_texts = [
            self.processor.apply_chat_template(
                self._messages(example, build_target(example)),
                tokenize=False,
                add_generation_prompt=False,
            )
            for example in examples
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                self._messages(example),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in examples
        ]
        images = [load_image(example.image_path) for example in examples]
        batch = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_batch = self.processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if input_ids.shape[1] > self.max_seq_length:
            raise ValueError(
                f"processed example exceeds max_seq_length={self.max_seq_length}; "
                "increase the bucket instead of truncating"
            )
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row, prompt_length in enumerate(prompt_lengths):
            if prompt_length > input_ids.shape[1]:
                raise ValueError("assistant boundary is outside the processed sequence")
            labels[row, : int(prompt_length)] = -100
        batch["labels"] = labels
        return {str(key): value for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--participant-json", type=Path, default=DEFAULT_PARTICIPANT_JSON)
    parser.add_argument("--participant-images", type=Path, default=DEFAULT_PARTICIPANT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/medreason/local_qwen_pilot"))
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
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-participant-inference", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"image reference is missing: {path}")
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def case_to_example(case: dict[str, Any], image_root: Path, *, labeled: bool) -> Example:
    case_id = str(case.get("case_id", "")).strip()
    task_type = str(case.get("question type", "")).strip().lower()
    question = str(case.get("question", "")).strip()
    if not case_id or not question or task_type not in {"mcq", "open-ended"}:
        raise ValueError("invalid case identity, task type, or question")
    raw_image = str(case.get("image_path", "")).strip()
    image_path = image_root / Path(raw_image).name
    if image_path.is_absolute() or ".." in image_path.relative_to(image_root).parts:
        raise ValueError("image reference escapes the configured image root")
    options = tuple((label, str(case.get(label, "")).strip()) for label in LABELS if str(case.get(label, "")).strip())
    answer_value = case.get("answer")
    answer = str(answer_value).strip() if answer_value is not None else None
    if task_type == "mcq":
        if len(options) < 2 or (labeled and answer not in {label for label, _ in options}):
            raise ValueError("MCQ case has invalid options or answer label")
    elif options:
        raise ValueError("open-ended case unexpectedly contains options")
    if labeled and answer is None:
        raise ValueError("labeled case has no answer")
    if not labeled and answer is not None:
        raise ValueError("participant-facing case contains an answer")
    return Example(case_id, task_type, question, options, answer, image_path)


def load_examples(json_path: Path, image_root: Path, *, labeled: bool) -> list[Example]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{json_path} does not contain a cases list")
    examples = [case_to_example(case, image_root, labeled=labeled) for case in cases]
    ids = [example.case_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{json_path} contains duplicate case IDs")
    return examples


def select_examples(examples: list[Example], limit: int) -> list[Example]:
    ordered = sorted(examples, key=lambda example: example.case_id)
    return ordered if limit <= 0 else ordered[:limit]


def split_train_dev(examples: list[Example], fraction: float, seed: int) -> tuple[list[Example], list[Example]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("dev fraction must be between zero and one")
    train: list[Example] = []
    dev: list[Example] = []
    for example in examples:
        digest = hashlib.sha256(f"medreason-local-dev-v1:{seed}:{example.case_id}".encode()).digest()
        is_dev = int.from_bytes(digest[:8], "big") / float(1 << 64) < fraction
        (dev if is_dev else train).append(example)
    if not train or not dev:
        raise ValueError("deterministic local split produced an empty partition")
    return train, dev


def build_prompt(example: Example) -> str:
    if example.task_type == "mcq":
        option_lines = "\n".join(f"{label}. {text}" for label, text in example.options)
        return (
            "Answer the medical question using the image. Return exactly the correct option label, "
            "a colon, and the option text. Do not provide reasoning.\n\n"
            f"Question: {example.question}\nOptions:\n{option_lines}"
        )
    return (
        "Answer the medical question using the image. Return one concise answer and do not provide "
        f"private reasoning.\n\nQuestion: {example.question}"
    )


def build_target(example: Example) -> str:
    if example.answer is None:
        raise ValueError("cannot build a target for an unlabeled example")
    if example.task_type == "mcq":
        option_text = dict(example.options)[example.answer]
        return f"{example.answer}: {option_text}"
    return example.answer


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def target_modules(model: torch.nn.Module) -> list[str]:
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in wanted
    }
    selected = sorted(found)
    if not selected:
        raise RuntimeError("could not discover Qwen LoRA target modules")
    return selected


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, torch.device, list[str]]:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("the local pilot requires CUDA; use the protected training path elsewhere")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modules = target_modules(model)
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


def optimizer_and_scheduler(model: torch.nn.Module, args: argparse.Namespace, steps: int) -> tuple[Any, Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    warmup = min(args.warmup_steps, max(0, steps - 1))

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / float(warmup)
        remaining = max(1, steps - warmup)
        return max(0.0, float(steps - step) / float(remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def evaluate_loss(model: Any, loader: DataLoader[dict[str, Tensor]], device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch)
            total_loss += float(output.loss.detach().float().cpu())
            batches += 1
    model.train()
    if batches == 0:
        raise ValueError("evaluation loader is empty")
    return total_loss / batches


def decode_generation(processor: Any, output_ids: Tensor, input_length: int) -> str:
    generated = output_ids[0, input_length:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_mcq_label(text: str) -> str | None:
    match = re.search(r"(?:^|[^A-Z])([A-E])\s*[:.)]", text.strip().upper())
    return match.group(1) if match else None


def generate_one(model: Any, processor: Any, example: Example, device: torch.device, max_new_tokens: int) -> str:
    messages = VisionLanguageCollator._messages(example)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[prompt], images=[load_image(example.image_path)], return_tensors="pt")
    batch = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}
    was_training = bool(model.training)
    previous_cache = getattr(model.config, "use_cache", False)
    model.eval()
    model.config.use_cache = True
    try:
        with torch.inference_mode():
            output_ids = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        model.config.use_cache = previous_cache
        model.train(was_training)
    return decode_generation(processor, output_ids, batch["input_ids"].shape[1])


def evaluate_generation(
    model: Any,
    processor: Any,
    examples: Iterable[Example],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    mcq_total = 0
    mcq_correct = 0
    open_total = 0
    outputs: list[dict[str, str]] = []
    for index, example in enumerate(examples, 1):
        raw = generate_one(model, processor, example, device, max_new_tokens)
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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
        json.dumps(
            {
                "train_source_cases": len(all_train),
                "selected_cases": len(selected),
                "train_cases": len(train_examples),
                "local_dev_cases": len(dev_examples),
                "participant_inference_cases": len(participant_examples),
            },
            sort_keys=True,
        ),
    )

    model, processor, device, modules = load_model_and_processor(args)
    collator = VisionLanguageCollator(processor, args.max_seq_length)
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
                    print(
                        "train_step",
                        json.dumps(
                            {"step": step, "loss": losses[-1], "peak_allocated_gib": round(peak, 3)}
                        ),
                    )
                if args.eval_every > 0 and step % args.eval_every == 0:
                    dev_loss = evaluate_loss(model, dev_loader, device)
                    print("dev_loss", json.dumps({"step": step, "loss": dev_loss}))
                if step >= total_steps:
                    break
        epoch += 1
    if microsteps % args.gradient_accumulation:
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
    dev_loss = evaluate_loss(model, dev_loader, device)
    dev_metrics = evaluate_generation(model, processor, dev_examples, device, args.max_new_tokens)

    model.config.use_cache = True
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    participant_records: list[dict[str, str]] = []
    if not args.skip_participant_inference:
        for index, example in enumerate(participant_examples, 1):
            raw = generate_one(model, processor, example, device, args.max_new_tokens)
            record = {
                "case_id": example.case_id,
                "task_type": example.task_type,
                "answer": parse_mcq_label(raw) if example.task_type == "mcq" else raw,
            }
            participant_records.append(record)
            if index % 16 == 0:
                print("participant_inference", index)
    save_json(args.output_dir / "participant_predictions.json", participant_records)
    summary = {
        "run_kind": "local_diagnostic_qwen2.5_vl_3b_adapter_pilot",
        "protected_medreason_route": False,
        "seed": args.seed,
        "model": str(args.model),
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
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": round(time.time() - start, 3),
    }
    save_json(args.output_dir / "local_dev_predictions.json", dev_metrics["outputs"])
    save_json(args.output_dir / "summary.json", summary)
    summary_for_log = {key: value for key, value in summary.items() if key != "lora_target_modules"}
    print("summary", json.dumps(summary_for_log, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
