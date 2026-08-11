#!/usr/bin/env python
"""Run a zero-shot MedReason baseline with a local pretrained VLM.

The script does not update model weights. It loads either the local Qwen2.5-VL
3B or Gemma 4 E4B checkpoint, evaluates a deterministic labeled probe from
Train, and writes challenge-format predictions for the participant-facing
validation split. Native MedReason JSON records and image folders are accepted
without repacking the dataset.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from scripts.medreason_data import Example, load_examples

LABELS = tuple("ABCDE")
DEFAULT_TRAIN_JSON = Path("../Data/Train/medreason_train_selection.json")
DEFAULT_TRAIN_IMAGES = Path("../Data/Train/imgs")
DEFAULT_TEST_JSON = Path("../Data/Valid/medreason_validation_participant_facing.json")
DEFAULT_TEST_IMAGES = Path("../Data/Valid/imgs")
DEFAULT_QWEN_MODEL = Path("artifacts/models/local/qwen2.5-vl-3b-instruct")
DEFAULT_GEMMA_MODEL = Path("artifacts/models/local/gemma-4-E4B-it")

SYSTEM_PROMPT = """You are a medical visual reasoning assistant for the MedReason challenge.
Use only evidence visible in the supplied image and the question. Do not invent
findings. Follow the task-specific output format in the user message. The
reasoning_trace, when requested, is a short evidence summary, not private
chain-of-thought."""


@dataclass(frozen=True)
class Prediction:
    case_id: str
    task_type: str
    answer: str
    reasoning_trace: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_type": self.task_type,
            "answer": self.answer,
            "reasoning_trace": self.reasoning_trace,
            "confidence": self.confidence,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "gemma"), default="gemma")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--test-json", "--valid-json", dest="test_json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--test-images", "--valid-images", dest="test_images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/medreason/pretrained_baseline"))
    parser.add_argument("--eval-limit", type=int, default=32, help="Labeled Train probe size; -1 uses all Train cases.")
    parser.add_argument("--test-limit", type=int, default=32, help="Validation inference size; -1 uses all cases.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--max-soft-tokens", type=int, default=70)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--no-4bit", action="store_true", help="Load weights in bfloat16 instead of 4-bit quantization."
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Validate data and output formatting without loading a model."
    )
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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


def stable_key(example: Example, seed: int) -> str:
    return hashlib.sha256(f"medreason-zero-shot-v1:{seed}:{example.case_id}".encode()).hexdigest()


def select_balanced(examples: Sequence[Example], limit: int, seed: int) -> list[Example]:
    """Select a deterministic subset while retaining both question formats."""
    ordered = sorted(examples, key=lambda example: stable_key(example, seed))
    if limit < 0 or limit >= len(ordered):
        return ordered
    if limit == 0:
        raise ValueError("case limits must be positive or -1 for all cases")

    groups: dict[str, list[Example]] = {"mcq": [], "open-ended": []}
    for example in ordered:
        groups.setdefault(example.task_type, []).append(example)
    if limit == 1 or len([group for group in groups.values() if group]) < 2:
        return ordered[:limit]

    quotas = {name: (limit * len(group)) // len(ordered) for name, group in groups.items() if group}
    for name in quotas:
        quotas[name] = max(1, quotas[name])
    while sum(quotas.values()) > limit:
        name = max(quotas, key=lambda key: quotas[key])
        if quotas[name] <= 1:
            break
        quotas[name] -= 1
    while sum(quotas.values()) < limit:
        name = max(quotas, key=lambda key: len(groups[name]) - quotas[key])
        if quotas[name] >= len(groups[name]):
            break
        quotas[name] += 1

    selected: list[Example] = []
    for name, quota in quotas.items():
        selected.extend(groups[name][:quota])
    return sorted(selected, key=lambda example: stable_key(example, seed))


def clean_option_text(label: str, text: str) -> str:
    prefix = rf"^\s*{re.escape(label)}\s*[\)\].:\-]\s*"
    return re.sub(prefix, "", text.strip(), count=1, flags=re.IGNORECASE)


def format_options(example: Example) -> str:
    return "\n".join(f"{label}. {clean_option_text(label, text)}" for label, text in example.options)


def build_prompt(example: Example) -> str:
    if example.task_type == "mcq":
        return f"""Task: closed-ended medical visual question answering.

Question:
{example.question}

Options:
{format_options(example)}

Inspect the image and choose the single best option. Output exactly one
uppercase option label from the provided options: A, B, C, D, or E. Output no
JSON, explanation, punctuation, or option text. Do not default to the first
option; decide from the image and question."""
    return f"""Task: open-ended medical visual question answering.

Question:
{example.question}

Inspect the image and answer the question. Return one valid JSON object with
exactly two string fields in this order: answer, reasoning_trace. Put the
concise final answer first so it is never omitted. Keep answer to at most 15
words and reasoning_trace to at most 20 words. The reasoning_trace must name
visible evidence supporting the answer. Do not include placeholders, markdown,
field descriptions, or private chain-of-thought. State a limitation when the
image cannot support a confident claim."""


def make_messages(example: Example, image_key: str) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [{"type": "image", image_key: str(example.image_path)}]
    content.append({"type": "text", "text": build_prompt(example)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def normalize_mcq_answer(raw_answer: str, example: Example) -> tuple[str, bool]:
    labels = tuple(label for label, _ in example.options)
    lookup = {label.upper(): label for label in labels}
    text = str(raw_answer).strip()
    if text.upper() in lookup:
        return lookup[text.upper()], True

    patterns = (
        r"(?:answer|option|choice|selected)\s*(?:is|:|-)?\s*[\(\[]?([A-E])\b",
        r"(?<![A-Za-z])([A-E])\s*[\)\].:,]",
        r"^\s*([A-E])\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(1).upper() in lookup:
            return lookup[match.group(1).upper()], True

    lowered = normalize_text(text)
    for label, option_text in example.options:
        cleaned = normalize_text(clean_option_text(label, option_text))
        if cleaned and cleaned in lowered:
            return label, True
    return labels[0], False


def parse_prediction(example: Example, raw_text: str) -> Prediction:
    obj = extract_json_object(raw_text) or {}
    raw_answer = str(obj.get("answer", obj.get("final_answer", ""))).strip()
    reasoning = str(obj.get("reasoning_trace", obj.get("rationale", ""))).strip()
    if not raw_answer:
        json_answer_match = re.search(r"""["']?answer["']?\s*:\s*["']([^"'\n}]*)""", raw_text, flags=re.IGNORECASE)
        if json_answer_match:
            raw_answer = json_answer_match.group(1).strip()
        else:
            answer_match = re.search(r"(?:final\s+answer|answer)\s*[:：]\s*(.+)", raw_text, flags=re.IGNORECASE)
            raw_answer = answer_match.group(1).strip().splitlines()[0] if answer_match else raw_text.strip()

    if example.task_type == "mcq":
        answer, parsed = normalize_mcq_answer(raw_answer, example)
        if not reasoning:
            reasoning = raw_text.strip() or f"Selected option {answer}."
        return Prediction(example.case_id, "mcq", answer, reasoning, 1.0 if parsed else 0.0)

    answer = raw_answer or "Unable to determine from the available visual evidence."
    if not reasoning:
        json_reasoning_match = re.search(
            r"""["']?reasoning_trace["']?\s*:\s*["']([^"'\n}]*)""",
            raw_text,
            flags=re.IGNORECASE,
        )
        if json_reasoning_match:
            reasoning = json_reasoning_match.group(1).strip()
        else:
            reasoning_match = re.search(r"reasoning(?:_trace)?\s*[:：]\s*(.+)", raw_text, flags=re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip().splitlines()[0] if reasoning_match else raw_text.strip()
    reasoning = reasoning or "The available visual evidence was insufficient for a reliable conclusion."
    return Prediction(example.case_id, "open", answer, reasoning, 1.0 if raw_answer else 0.0)


class VLMRunner:
    def generate(self, example: Example) -> str:
        raise NotImplementedError


class QwenRunner(VLMRunner):
    def __init__(self, args: argparse.Namespace, model_path: Path) -> None:
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        self.device = torch.device("cuda:0")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
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
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

    def generate(self, example: Example) -> str:
        messages = make_messages(example, "image")
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(
            text=[prompt],
            images=[load_image(example.image_path)],
            padding=True,
            return_tensors="pt",
        )
        batch = move_tensors(batch, self.device)
        previous_cache = getattr(self.model.config, "use_cache", False)
        self.model.config.use_cache = True
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False)
        finally:
            self.model.config.use_cache = previous_cache
        prompt_length = batch["input_ids"].shape[1]
        generated = output_ids[:, prompt_length:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0].strip()

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value: int) -> None:
        self._max_new_tokens = value


class GemmaRunner(VLMRunner):
    def __init__(self, args: argparse.Namespace, model_path: Path) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

        self.device = torch.device("cuda:0")
        self.processor = AutoProcessor.from_pretrained(model_path)
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
        self.model = AutoModelForMultimodalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        self.max_soft_tokens = args.max_soft_tokens
        self._set_cache(True)

    def _set_cache(self, enabled: bool) -> None:
        configs = [
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "config", None), "text_config", None),
        ]
        for config in configs:
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = enabled

    def generate(self, example: Example) -> str:
        messages = make_messages(example, "url")
        processor_kwargs = {"images_kwargs": {"max_soft_tokens": self.max_soft_tokens}}
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            return_assistant_tokens_mask=False,
            enable_thinking=False,
            processor_kwargs=processor_kwargs,
        )
        batch = move_tensors(batch, self.device)
        self._set_cache(True)
        with torch.inference_mode():
            output_ids = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output_ids[:, batch["input_ids"].shape[1] :]
        try:
            parsed = self.processor.parse_response(generated[0], prefix=batch["input_ids"][0])
            if isinstance(parsed, dict) and str(parsed.get("content", "")).strip():
                return str(parsed["content"]).strip()
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        return self.processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value: int) -> None:
        self._max_new_tokens = value


def build_runner(args: argparse.Namespace) -> VLMRunner:
    if not torch.cuda.is_available():
        raise RuntimeError("pretrained baseline inference requires CUDA; use --smoke for data-only validation")
    model_path = args.model_path or (DEFAULT_QWEN_MODEL if args.model == "qwen" else DEFAULT_GEMMA_MODEL)
    if not model_path.is_dir():
        raise FileNotFoundError(f"local pretrained checkpoint directory does not exist: {model_path}")
    runner: VLMRunner
    if args.model == "qwen":
        runner = QwenRunner(args, model_path)
    else:
        runner = GemmaRunner(args, model_path)
    runner.max_new_tokens = args.max_new_tokens  # type: ignore[attr-defined]
    return runner


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens or not target_tokens:
        return float(pred_tokens == target_tokens)
    pred_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1
    overlap = sum(min(count, target_counts.get(token, 0)) for token, count in pred_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def infer_cases(
    runner: VLMRunner, examples: Iterable[Example], *, labeled: bool
) -> tuple[list[Prediction], list[dict[str, Any]]]:
    predictions: list[Prediction] = []
    records: list[dict[str, Any]] = []
    examples_list = list(examples)
    for index, example in enumerate(examples_list, start=1):
        raw = runner.generate(example)
        prediction = parse_prediction(example, raw)
        predictions.append(prediction)
        record = prediction.to_dict()
        record["raw_output"] = raw
        if labeled:
            record["target_answer"] = example.answer
            record["mcq_correct"] = prediction.answer == example.answer if example.task_type == "mcq" else None
            record["open_token_f1"] = (
                token_f1(prediction.answer, example.answer or "") if example.task_type != "mcq" else None
            )
        records.append(record)
        print(f"inference {index}/{len(examples_list)} {example.case_id}", flush=True)
    return predictions, records


def summarize_labeled(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    mcq = [record for record in records if record["task_type"] == "mcq"]
    open_records = [record for record in records if record["task_type"] == "open"]
    fallback_count = sum(float(record["confidence"]) == 0.0 for record in records)
    return {
        "cases": len(records),
        "mcq_cases": len(mcq),
        "mcq_correct": sum(bool(record["mcq_correct"]) for record in mcq),
        "mcq_accuracy": (sum(bool(record["mcq_correct"]) for record in mcq) / len(mcq)) if mcq else None,
        "open_cases": len(open_records),
        "open_exact_match": (
            sum(
                normalize_text(str(record["answer"])) == normalize_text(str(record["target_answer"]))
                for record in open_records
            )
            / len(open_records)
            if open_records
            else None
        ),
        "open_token_f1": (
            sum(float(record["open_token_f1"]) for record in open_records) / len(open_records) if open_records else None
        ),
        "parser_fallbacks": fallback_count,
    }


def validate_predictions(predictions: Sequence[Prediction], examples: Sequence[Example]) -> None:
    if len(predictions) != len(examples):
        raise ValueError(f"prediction count {len(predictions)} does not match case count {len(examples)}")
    expected = {example.case_id: example for example in examples}
    seen: set[str] = set()
    for prediction in predictions:
        if prediction.case_id in seen:
            raise ValueError(f"duplicate prediction for case_id={prediction.case_id}")
        seen.add(prediction.case_id)
        example = expected.get(prediction.case_id)
        if example is None:
            raise ValueError(f"prediction contains unknown case_id={prediction.case_id}")
        expected_task = "mcq" if example.task_type == "mcq" else "open"
        if prediction.task_type != expected_task:
            raise ValueError(f"task_type mismatch for {prediction.case_id}")
        if not prediction.answer.strip():
            raise ValueError(f"empty answer for {prediction.case_id}")
        if expected_task == "mcq" and prediction.answer not in {label for label, _ in example.options}:
            raise ValueError(f"invalid MCQ answer for {prediction.case_id}: {prediction.answer!r}")
        if expected_task == "open" and not prediction.reasoning_trace.strip():
            raise ValueError(f"open-ended case {prediction.case_id} has no reasoning trace")
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ValueError(f"missing predictions for case IDs: {missing[:3]}")


def results_payload(predictions: Sequence[Prediction], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "Medical visual reasoning",
        "answers": [prediction.to_dict() for prediction in predictions],
        "version": {"major": 1, "minor": 0},
    }


def smoke_predictions(examples: Sequence[Example]) -> list[Prediction]:
    predictions: list[Prediction] = []
    for example in examples:
        if example.task_type == "mcq":
            answer = example.options[0][0]
            predictions.append(
                Prediction(
                    example.case_id,
                    "mcq",
                    answer,
                    "Smoke mode; no model inference was performed.",
                    0.0,
                )
            )
        else:
            predictions.append(
                Prediction(
                    example.case_id,
                    "open",
                    "Unable to determine from the available visual evidence.",
                    "Smoke mode; no model inference was performed.",
                    0.0,
                )
            )
    return predictions


def main() -> int:
    args = parse_args()
    if args.eval_limit == 0 or args.test_limit == 0:
        raise ValueError("case limits must be positive or -1 for all cases")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    train_examples = load_examples(args.train_json, args.train_images, labeled=True)
    test_examples = load_examples(args.test_json, args.test_images, labeled=False)
    eval_examples = select_balanced(train_examples, args.eval_limit, args.seed)
    selected_test = select_balanced(test_examples, args.test_limit, args.seed + 1)
    print(
        json.dumps(
            {
                "train_cases": len(train_examples),
                "test_cases": len(test_examples),
                "eval_probe_cases": len(eval_examples),
                "test_inference_cases": len(selected_test),
                "model": args.model,
                "smoke": args.smoke,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.smoke:
        eval_predictions = smoke_predictions(eval_examples) if not args.skip_eval else []
        test_predictions = smoke_predictions(selected_test) if not args.skip_test else []
        eval_records = []
        for prediction, example in zip(eval_predictions, eval_examples, strict=True):
            record = prediction.to_dict()
            record["target_answer"] = example.answer
            record["mcq_correct"] = prediction.answer == example.answer if example.task_type == "mcq" else None
            record["open_token_f1"] = (
                token_f1(prediction.answer, example.answer or "") if example.task_type != "mcq" else None
            )
            eval_records.append(record)
    else:
        runner = build_runner(args)
        eval_predictions, eval_records = (
            infer_cases(runner, eval_examples, labeled=True) if not args.skip_eval else ([], [])
        )
        test_predictions, _ = infer_cases(runner, selected_test, labeled=False) if not args.skip_test else ([], [])

    if eval_predictions:
        validate_predictions(eval_predictions, eval_examples)
    if test_predictions:
        validate_predictions(test_predictions, selected_test)
    if eval_records:
        save_json(args.output_dir / "labeled_probe_predictions.json", eval_records)
    if test_predictions:
        save_json(
            args.output_dir / "results.json",
            results_payload(test_predictions, f"MedReason {args.model} pretrained zero-shot baseline predictions"),
        )

    summary: dict[str, Any] = {
        "run_kind": "medreason_pretrained_zero_shot_baseline",
        "model": args.model,
        "model_path": str(args.model_path or (DEFAULT_QWEN_MODEL if args.model == "qwen" else DEFAULT_GEMMA_MODEL)),
        "four_bit": not args.no_4bit,
        "seed": args.seed,
        "train_json": str(args.train_json),
        "test_json": str(args.test_json),
        "train_json_sha256": sha256_file(args.train_json),
        "test_json_sha256": sha256_file(args.test_json),
        "train_cases": len(train_examples),
        "test_cases": len(test_examples),
        "eval_probe_cases": len(eval_examples),
        "test_inference_cases": len(selected_test),
        "eval_skipped": args.skip_eval,
        "test_skipped": args.skip_test,
        "smoke": args.smoke,
        "labeled_probe": summarize_labeled(eval_records) if eval_records else None,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    if torch.cuda.is_available():
        summary["gpu"] = torch.cuda.get_device_name(0)
        summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, default=str), flush=True)

    if not args.smoke:
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
