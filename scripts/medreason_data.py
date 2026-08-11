"""Deterministic MedReason data loading and Unsloth conversation formatting.

The dataset contract is intentionally model-agnostic.  Model-specific training
code consumes :class:`Example` objects and turns them into the multimodal
``messages`` format expected by Unsloth's vision data collator.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

LABELS = tuple("ABCDE")
DEFAULT_TRAIN_JSON = Path("../Data/Train/medreason_train_selection.json")
DEFAULT_TRAIN_IMAGES = Path("../Data/Train/imgs")
DEFAULT_PARTICIPANT_JSON = Path("../Data/Valid/medreason_validation_participant_facing.json")
DEFAULT_PARTICIPANT_IMAGES = Path("../Data/Valid/imgs")


@dataclass(frozen=True)
class Example:
    """One labeled or participant-facing 2D visual question-answer case."""

    case_id: str
    task_type: str
    question: str
    options: tuple[tuple[str, str], ...]
    answer: str | None
    image_path: Path


def load_image(path: Path) -> Image.Image:
    """Load an image eagerly and detach it from the source file handle."""
    if not path.is_file():
        raise FileNotFoundError(f"image reference is missing: {path}")
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def _safe_image_path(image_root: Path, raw_image: str) -> Path:
    root = image_root.expanduser().resolve()
    candidate = (root / raw_image).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("image reference escapes the configured image root")
    return candidate


def case_to_example(case: Mapping[str, Any], image_root: Path, *, labeled: bool) -> Example:
    case_id = str(case.get("case_id", "")).strip()
    task_type = str(case.get("question type", "")).strip().lower()
    question = str(case.get("question", "")).strip()
    if not case_id or not question or task_type not in {"mcq", "open-ended"}:
        raise ValueError("invalid case identity, task type, or question")

    raw_image = str(case.get("image_path", "")).strip()
    if not raw_image:
        raise ValueError(f"case {case_id!r} has no image_path")
    image_path = _safe_image_path(image_root, raw_image)

    options = tuple((label, str(case.get(label, "")).strip()) for label in LABELS if str(case.get(label, "")).strip())
    answer_value = case.get("answer")
    answer = str(answer_value).strip() if answer_value is not None else None
    if task_type == "mcq":
        option_labels = {label for label, _ in options}
        if len(options) < 2 or (labeled and answer not in option_labels):
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


def select_examples(examples: Iterable[Example], limit: int) -> list[Example]:
    ordered = sorted(examples, key=lambda example: example.case_id)
    return ordered if limit <= 0 else ordered[:limit]


def split_train_dev(examples: Iterable[Example], fraction: float, seed: int) -> tuple[list[Example], list[Example]]:
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


def load_and_select_examples(
    *,
    train_json: Path,
    train_images: Path,
    participant_json: Path,
    participant_images: Path,
    train_limit: int,
    dev_limit: int,
    participant_limit: int,
    dev_fraction: float,
    seed: int,
) -> tuple[list[Example], list[Example], list[Example], list[Example], list[Example]]:
    if train_limit == 0 or dev_limit == 0 or participant_limit == 0:
        raise ValueError("limits must be positive, or negative for all examples")
    all_train = load_examples(train_json, train_images, labeled=True)
    selected = select_examples(all_train, train_limit)
    train_examples, dev_examples = split_train_dev(selected, dev_fraction, seed)
    dev_examples = select_examples(dev_examples, dev_limit)
    participant_examples = select_examples(
        load_examples(participant_json, participant_images, labeled=False), participant_limit
    )
    return all_train, selected, train_examples, dev_examples, participant_examples


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


def build_messages(
    example: Example,
    *,
    target: str | None = None,
    image: Image.Image | str | None = None,
) -> list[dict[str, Any]]:
    """Build the official Unsloth multimodal conversation shape.

    Training records use an eagerly loaded RGB PIL image.  Inference callers
    may provide a path/string when their processor resolves image references.
    """
    image_value: Image.Image | str = image if image is not None else str(example.image_path)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_prompt(example)},
        {"type": "image", "image": image_value},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def to_unsloth_record(example: Example) -> dict[str, Any]:
    """Convert one labeled example to an Unsloth SFT record."""
    return {
        "messages": build_messages(
            example,
            target=build_target(example),
            image=load_image(example.image_path),
        )
    }


def to_unsloth_records(examples: Iterable[Example]) -> list[dict[str, Any]]:
    return [to_unsloth_record(example) for example in examples]


def parse_mcq_label(text: str) -> str | None:
    import re

    match = re.search(r"(?:^|[^A-Z])([A-E])\s*[:.)]", text.strip().upper())
    return match.group(1) if match else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_PARTICIPANT_IMAGES",
    "DEFAULT_PARTICIPANT_JSON",
    "DEFAULT_TRAIN_IMAGES",
    "DEFAULT_TRAIN_JSON",
    "Example",
    "LABELS",
    "build_messages",
    "build_prompt",
    "build_target",
    "case_to_example",
    "load_and_select_examples",
    "load_examples",
    "load_image",
    "parse_mcq_label",
    "save_json",
    "select_examples",
    "sha256_file",
    "split_train_dev",
    "to_unsloth_record",
    "to_unsloth_records",
]
