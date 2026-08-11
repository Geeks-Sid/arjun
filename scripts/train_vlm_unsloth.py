#!/usr/bin/env python
"""Unsloth/TRL vision-language SFT runner for 2D medical images.

This is the only production 2D VLM training loop in the repository.  It keeps
medical data preparation local to the project, while Unsloth owns model
patching, PEFT injection, multimodal collation, optimizer setup, and the
training loop through ``trl.SFTTrainer``.

The optional ``unsloth`` dependency is imported only after data validation, so
``--dry-run`` and the CPU contract tests do not require CUDA kernels.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from scripts.medreason_data import (
    DEFAULT_PARTICIPANT_IMAGES,
    DEFAULT_PARTICIPANT_JSON,
    DEFAULT_TRAIN_IMAGES,
    DEFAULT_TRAIN_JSON,
    Example,
    build_messages,
    load_and_select_examples,
    load_image,
    parse_mcq_label,
    save_json,
    to_unsloth_records,
)


class UnslothVLMError(RuntimeError):
    """Raised when the optional Unsloth VLM runtime cannot be constructed."""


@dataclass(frozen=True)
class UnslothVLMConfig:
    """Resolved settings shared by Qwen-VL, Gemma-Vision, and compatible VLMs."""

    model: str
    output_dir: Path
    train_examples: tuple[Example, ...]
    eval_examples: tuple[Example, ...]
    participant_examples: tuple[Example, ...] = ()
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_steps: int = 100
    epochs: int = 1
    logging_steps: int = 1
    eval_steps: int = 50
    save_steps: int = 50
    seed: int = 2026
    load_in_4bit: bool = True
    use_gradient_checkpointing: str | bool = "unsloth"
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    image_min_pixels: int | None = None
    image_max_pixels: int | None = None
    max_new_tokens: int = 64
    skip_participant_inference: bool = False
    report_to: str = "none"
    resume_from_checkpoint: str | None = None
    model_family: str = "generic-vlm"
    terms_url: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.train_examples:
            raise ValueError("at least one training example is required")
        if not self.eval_examples:
            raise ValueError("at least one evaluation example is required")
        for name in (
            "max_seq_length",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "warmup_steps",
            "epochs",
            "logging_steps",
            "eval_steps",
            "save_steps",
            "lora_rank",
            "lora_alpha",
            "max_new_tokens",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be positive, or -1 to derive it from epochs")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not any((self.finetune_vision_layers, self.finetune_language_layers)):
            raise ValueError("at least one of vision or language layers must be fine-tuned")
        if self.image_min_pixels is not None and self.image_min_pixels <= 0:
            raise ValueError("image_min_pixels must be positive")
        if self.image_max_pixels is not None and self.image_max_pixels <= 0:
            raise ValueError("image_max_pixels must be positive")
        if (
            self.image_min_pixels is not None
            and self.image_max_pixels is not None
            and self.image_min_pixels > self.image_max_pixels
        ):
            raise ValueError("image_min_pixels cannot exceed image_max_pixels")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["train_examples"] = len(self.train_examples)
        payload["eval_examples"] = len(self.eval_examples)
        payload["participant_examples"] = len(self.participant_examples)
        return payload


def _accepted_kwargs(factory: Callable[..., Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Keep one runner compatible with the small TRL/Unsloth API drift."""
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return dict(values)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(values)
    return {name: value for name, value in values.items() if name in parameters}


def _configure_image_processor(processor: Any, config: UnslothVLMConfig) -> None:
    requested = {
        "min_pixels": config.image_min_pixels,
        "max_pixels": config.image_max_pixels,
    }
    requested = {name: value for name, value in requested.items() if value is not None}
    if not requested:
        return
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise UnslothVLMError("the selected VLM processor does not expose image pixel bounds")
    missing: list[str] = []
    for name, value in requested.items():
        if not hasattr(image_processor, name):
            missing.append(name)
            continue
        setattr(image_processor, name, value)
    if missing:
        names = ", ".join(missing)
        raise UnslothVLMError(f"the selected VLM processor does not support image bounds: {names}")


def _set_alias(
    values: dict[str, Any],
    parameters: Mapping[str, inspect.Parameter],
    names: Sequence[str],
    value: Any,
) -> None:
    for name in names:
        if name in parameters:
            values[name] = value
            return


def _load_unsloth_model(config: UnslothVLMConfig) -> tuple[Any, Any, Any]:
    if not torch.cuda.is_available():
        raise UnslothVLMError("Unsloth VLM fine-tuning requires CUDA; use --dry-run for data-only validation")
    try:
        from unsloth import FastVisionModel
    except ImportError as exc:
        raise UnslothVLMError(
            "Unsloth is not installed; run `uv sync --extra vlm` (or `make install-vlm)` before training"
        ) from exc

    load_values: dict[str, Any] = {
        "model_name": config.model,
        "load_in_4bit": config.load_in_4bit,
        "use_gradient_checkpointing": config.use_gradient_checkpointing,
        "max_seq_length": config.max_seq_length,
    }
    model, processor = FastVisionModel.from_pretrained(**_accepted_kwargs(FastVisionModel.from_pretrained, load_values))
    _configure_image_processor(processor, config)

    peft_values = {
        "finetune_vision_layers": config.finetune_vision_layers,
        "finetune_language_layers": config.finetune_language_layers,
        "finetune_attention_modules": config.finetune_attention_modules,
        "finetune_mlp_modules": config.finetune_mlp_modules,
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": "none",
        "random_state": config.seed,
        "use_rslora": False,
        "loftq_config": None,
        "target_modules": "all-linear",
    }
    model = FastVisionModel.get_peft_model(
        model,
        **_accepted_kwargs(FastVisionModel.get_peft_model, peft_values),
    )
    return model, processor, FastVisionModel


def _build_collator(model: Any, processor: Any, config: UnslothVLMConfig) -> Any:
    try:
        from unsloth.trainer import UnslothVisionDataCollator
    except ImportError as exc:
        raise UnslothVLMError(
            "the installed Unsloth package does not provide UnslothVisionDataCollator; upgrade the `vlm` extra"
        ) from exc
    values = {
        "max_seq_length": config.max_seq_length,
        "completion_only_loss": True,
    }
    return UnslothVisionDataCollator(
        model,
        processor,
        **_accepted_kwargs(UnslothVisionDataCollator, values),
    )


def _build_sft_config(SFTConfig: Callable[..., Any], config: UnslothVLMConfig, *, has_eval: bool) -> Any:
    try:
        parameters = inspect.signature(SFTConfig).parameters
    except (TypeError, ValueError):
        parameters = {}
    values: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_steps": config.warmup_steps,
        "max_steps": config.max_steps,
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "logging_steps": config.logging_steps,
        "logging_first_step": True,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 2,
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit" if config.load_in_4bit else "adamw_torch_fused",
        "seed": config.seed,
        "report_to": config.report_to,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "packing": False,
        "gradient_checkpointing": True,
        "bf16": True,
        "fp16": False,
    }
    _set_alias(values, parameters, ("eval_strategy", "evaluation_strategy"), "steps" if has_eval else "no")
    _set_alias(values, parameters, ("max_seq_length", "max_length"), config.max_seq_length)
    _set_alias(values, parameters, ("save_strategy",), "steps")
    if not has_eval:
        values.pop("eval_steps", None)
    if not parameters:
        return SFTConfig(**values)
    return SFTConfig(**{name: value for name, value in values.items() if name in parameters})


def _build_trainer(
    SFTTrainer: Callable[..., Any],
    *,
    model: Any,
    processor: Any,
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]],
    data_collator: Any,
    training_args: Any,
) -> Any:
    try:
        parameters = inspect.signature(SFTTrainer).parameters
    except (TypeError, ValueError):
        parameters = {}
    values: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "args": training_args,
    }
    if "processing_class" in parameters:
        values["processing_class"] = processor
    else:
        values["tokenizer"] = processor
    if not parameters:
        return SFTTrainer(**values)
    return SFTTrainer(**{name: value for name, value in values.items() if name in parameters})


def build_unsloth_trainer(config: UnslothVLMConfig) -> tuple[Any, Any, Any]:
    """Load the model and construct the response-only multimodal SFT trainer."""
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise UnslothVLMError("TRL is required by Unsloth's VLM trainer; install the `hf` and `vlm` extras") from exc

    train_dataset = to_unsloth_records(config.train_examples)
    eval_dataset = to_unsloth_records(config.eval_examples)
    model, processor, fast_vision_model = _load_unsloth_model(config)
    collator = _build_collator(model, processor, config)
    training_args = _build_sft_config(SFTConfig, config, has_eval=bool(eval_dataset))
    trainer = _build_trainer(
        SFTTrainer,
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        training_args=training_args,
    )
    # Keep the class available for the optional post-train generation pass
    # without making it part of the SFTTrainer public contract.
    trainer._unsloth_fast_vision_model = fast_vision_model
    return model, processor, trainer


def set_model_cache(model: Any, enabled: bool) -> None:
    """Set cache flags on both top-level and nested text configs when present."""
    configs = [getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)]
    for model_config in configs:
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = enabled


def _device_for_model(model: Any) -> torch.device:
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        return torch.device("cuda:0")
    return parameter.device


def _processor_inputs(processor: Any, messages: list[dict[str, Any]], image: Image.Image) -> dict[str, Any]:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        return dict(processor(text=[prompt], images=[image], return_tensors="pt"))
    except (TypeError, ValueError):
        try:
            return dict(processor(image, prompt, add_special_tokens=False, return_tensors="pt"))
        except (TypeError, ValueError):
            return dict(
                processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
            )


def generate_one(model: Any, processor: Any, example: Example, max_new_tokens: int) -> str:
    """Generate one bounded answer using the same conversation contract as SFT."""
    image = load_image(example.image_path)
    messages = build_messages(example, image=image)
    batch = _processor_inputs(processor, messages, image)
    device = _device_for_model(model)
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    was_training = bool(getattr(model, "training", False))
    previous_cache = getattr(getattr(model, "config", None), "use_cache", False)
    model.eval()
    set_model_cache(model, True)
    try:
        with torch.inference_mode():
            output_ids = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        set_model_cache(model, previous_cache)
        model.train(was_training)
    input_length = int(batch["input_ids"].shape[-1])
    generated = output_ids[0, input_length:]
    tokenizer = getattr(processor, "tokenizer", processor)
    return str(tokenizer.decode(generated.tolist(), skip_special_tokens=True)).strip()


def evaluate_generation(
    model: Any,
    processor: Any,
    examples: Iterable[Example],
    max_new_tokens: int,
) -> dict[str, Any]:
    mcq_total = 0
    mcq_correct = 0
    open_total = 0
    outputs: list[dict[str, str]] = []
    for example in examples:
        generated = generate_one(model, processor, example, max_new_tokens)
        if example.task_type == "mcq":
            predicted = parse_mcq_label(generated)
            mcq_total += 1
            mcq_correct += int(predicted == example.answer)
            outputs.append(
                {
                    "case_id": example.case_id,
                    "predicted_label": predicted or "",
                    "target_label": example.answer or "",
                    "generated": generated,
                }
            )
        else:
            open_total += 1
            outputs.append({"case_id": example.case_id, "generated": generated})
    return {
        "mcq_cases": mcq_total,
        "mcq_correct": mcq_correct,
        "mcq_accuracy": (mcq_correct / mcq_total) if mcq_total else None,
        "open_cases": open_total,
        "outputs": outputs,
    }


def _run_participant_inference(
    model: Any,
    processor: Any,
    examples: Iterable[Example],
    max_new_tokens: int,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for example in examples:
        generated = generate_one(model, processor, example, max_new_tokens)
        records.append(
            {
                "case_id": example.case_id,
                "task_type": example.task_type,
                "answer": parse_mcq_label(generated) or "" if example.task_type == "mcq" else generated,
                "generated": generated,
            }
        )
    return records


def train_unsloth_vlm(config: UnslothVLMConfig) -> dict[str, Any]:
    """Run SFT, save the adapter/processor, and write a reproducible summary."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, processor, trainer = build_unsloth_trainer(config)
    train_call = {"resume_from_checkpoint": config.resume_from_checkpoint} if config.resume_from_checkpoint else {}
    train_output = trainer.train(**train_call)
    trainer.save_model(str(config.output_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(config.output_dir))

    log_history = list(getattr(getattr(trainer, "state", None), "log_history", []))
    participant_predictions: list[dict[str, str]] = []
    dev_metrics: dict[str, Any] = {}
    if not config.skip_participant_inference:
        fast_vision_model = getattr(trainer, "_unsloth_fast_vision_model", None)
        if fast_vision_model is not None and hasattr(fast_vision_model, "for_inference"):
            fast_vision_model.for_inference(model)
        dev_metrics = evaluate_generation(model, processor, config.eval_examples, config.max_new_tokens)
        participant_predictions = _run_participant_inference(
            model,
            processor,
            config.participant_examples,
            config.max_new_tokens,
        )

    train_metrics = dict(getattr(train_output, "metrics", {}) or {})
    summary: dict[str, Any] = {
        "training_started": True,
        "trainer_backend": "unsloth",
        "trainer_class": type(trainer).__name__,
        "model_family": config.model_family,
        "terms_url": config.terms_url,
        "model": config.model,
        "train_cases": len(config.train_examples),
        "eval_cases": len(config.eval_examples),
        "participant_inference_cases": len(config.participant_examples),
        "optimizer_steps": int(getattr(getattr(trainer, "state", None), "global_step", 0)),
        "train_metrics": train_metrics,
        "local_dev_generation": {key: value for key, value in dev_metrics.items() if key != "outputs"},
        "participant_labels_available": False,
        "load_in_4bit": config.load_in_4bit,
        "finetune_vision_layers": config.finetune_vision_layers,
        "finetune_language_layers": config.finetune_language_layers,
        "finetune_attention_modules": config.finetune_attention_modules,
        "finetune_mlp_modules": config.finetune_mlp_modules,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "max_seq_length": config.max_seq_length,
        "elapsed_seconds": round(time.time() - started, 3),
        "log_history_tail": log_history[-10:],
    }
    save_json(config.output_dir / "local_dev_predictions.json", dev_metrics.get("outputs", []))
    save_json(config.output_dir / "participant_predictions.json", participant_predictions)
    save_json(config.output_dir / "run_config.json", config.to_dict())
    save_json(config.output_dir / "summary.json", summary)
    return summary


def _dry_run_plan(
    args: argparse.Namespace,
    *,
    model_family: str,
    terms_url: str | None,
    all_train: Sequence[Example],
    selected: Sequence[Example],
    train_examples: Sequence[Example],
    eval_examples: Sequence[Example],
    participant_examples: Sequence[Example],
) -> dict[str, Any]:
    return {
        "training_started": False,
        "trainer_backend": "unsloth",
        "model_family": model_family,
        "terms_url": terms_url,
        "model": str(args.model),
        "train_source_cases": len(all_train),
        "selected_cases": len(selected),
        "train_cases": len(train_examples),
        "local_dev_cases": len(eval_examples),
        "participant_inference_cases": len(participant_examples),
        "lora_scope": {
            "vision_layers": bool(args.finetune_vision_layers),
            "language_layers": not bool(args.no_finetune_language_layers),
            "attention_modules": not bool(args.no_finetune_attention_modules),
            "mlp_modules": not bool(args.no_finetune_mlp_modules),
        },
        "four_bit": not bool(args.no_4bit),
        "max_seq_length": int(args.max_seq_length),
        "gradient_accumulation": int(args.gradient_accumulation),
    }


def run_cli(args: argparse.Namespace, *, model_family: str, terms_url: str | None = None) -> int:
    all_train, selected, train_examples, eval_examples, participant_examples = load_and_select_examples(
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
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_plan(
                    args,
                    model_family=model_family,
                    terms_url=terms_url,
                    all_train=all_train,
                    selected=selected,
                    train_examples=train_examples,
                    eval_examples=eval_examples,
                    participant_examples=participant_examples,
                ),
                sort_keys=True,
            )
        )
        return 0

    config = UnslothVLMConfig(
        model=str(args.model),
        output_dir=args.output_dir,
        train_examples=tuple(train_examples),
        eval_examples=tuple(eval_examples),
        participant_examples=tuple(participant_examples),
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_every,
        save_steps=args.save_every,
        seed=args.seed,
        load_in_4bit=not args.no_4bit,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=not args.no_finetune_language_layers,
        finetune_attention_modules=not args.no_finetune_attention_modules,
        finetune_mlp_modules=not args.no_finetune_mlp_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        image_min_pixels=args.min_pixels,
        image_max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        skip_participant_inference=args.skip_participant_inference,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume,
        model_family=model_family,
        terms_url=terms_url,
    )
    summary = train_unsloth_vlm(config)
    print("summary", json.dumps(summary, sort_keys=True, default=str))
    return 0


def build_parser(
    *,
    description: str,
    default_model: str,
    default_output_dir: Path,
    model_family: str,
    default_min_pixels: int | None = None,
    default_max_pixels: int | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=default_model, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--participant-json", type=Path, default=DEFAULT_PARTICIPANT_JSON)
    parser.add_argument("--participant-images", type=Path, default=DEFAULT_PARTICIPANT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--dev-limit", type=int, default=32)
    parser.add_argument("--participant-limit", type=int, default=64)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int, default=default_min_pixels)
    parser.add_argument("--max-pixels", type=int, default=default_max_pixels)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume", help="checkpoint directory accepted by SFTTrainer")
    parser.add_argument("--no-4bit", action="store_true", help="load BF16 base weights instead of NF4")
    parser.add_argument(
        "--finetune-vision-layers",
        action="store_true",
        help="also add LoRA adapters to the vision tower; disabled by default for 2D PEFT runs",
    )
    parser.add_argument("--no-finetune-language-layers", action="store_true")
    parser.add_argument("--no-finetune-attention-modules", action="store_true")
    parser.add_argument("--no-finetune-mlp-modules", action="store_true")
    parser.add_argument("--skip-participant-inference", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate data and print the Unsloth plan without importing Unsloth or allocating model weights",
    )
    parser.set_defaults(model_family=model_family)
    return parser


def main() -> int:
    parser = build_parser(
        description=__doc__ or "Unsloth VLM fine-tuning",
        default_model="unsloth/Qwen2.5-VL-3B-Instruct",
        default_output_dir=Path("artifacts/runs/medreason/unsloth_vlm_adapter"),
        model_family="generic-vlm",
        default_min_pixels=256 * 28 * 28,
        default_max_pixels=512 * 28 * 28,
    )
    args = parser.parse_args()
    return run_cli(args, model_family="generic-vlm")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "UnslothVLMConfig",
    "UnslothVLMError",
    "build_parser",
    "build_unsloth_trainer",
    "evaluate_generation",
    "generate_one",
    "main",
    "run_cli",
    "set_model_cache",
    "train_unsloth_vlm",
]
