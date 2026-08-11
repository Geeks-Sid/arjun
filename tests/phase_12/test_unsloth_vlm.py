from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.medreason_data import Example, build_messages, case_to_example, to_unsloth_record
from scripts.train_vlm_unsloth import UnslothVLMConfig


def test_unsloth_builder_wires_response_only_sft_stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys
    import types

    import torch

    image_path = tmp_path / "case.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    example = _example(image_path)
    calls: dict[str, object] = {}

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakeImageProcessor:
        min_pixels = 1
        max_pixels = 2

    class FakeProcessor:
        def __init__(self) -> None:
            self.image_processor = FakeImageProcessor()

    class FakeFastVisionModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["load"] = kwargs
            return FakeModel(), FakeProcessor()

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["peft"] = kwargs
            return model

    class FakeCollator:
        def __init__(self, model, processor, **kwargs) -> None:
            calls["collator"] = kwargs

    class FakeSFTConfig:
        def __init__(self, output_dir, **kwargs) -> None:
            self.output_dir = output_dir
            self.kwargs = kwargs

    class FakeSFTTrainer:
        def __init__(self, model, train_dataset, eval_dataset, data_collator, args, processing_class) -> None:
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.data_collator = data_collator
            self.args = args
            self.processing_class = processing_class

    unsloth_module = types.ModuleType("unsloth")
    unsloth_module.FastVisionModel = FakeFastVisionModel
    unsloth_trainer_module = types.ModuleType("unsloth.trainer")
    unsloth_trainer_module.UnslothVisionDataCollator = FakeCollator
    trl_module = types.ModuleType("trl")
    trl_module.SFTConfig = FakeSFTConfig
    trl_module.SFTTrainer = FakeSFTTrainer
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_module)
    monkeypatch.setitem(sys.modules, "unsloth.trainer", unsloth_trainer_module)
    monkeypatch.setitem(sys.modules, "trl", trl_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    from scripts.train_vlm_unsloth import UnslothVLMConfig, build_unsloth_trainer

    config = UnslothVLMConfig(
        model="test-vlm",
        output_dir=tmp_path / "out",
        train_examples=(example,),
        eval_examples=(example,),
        image_min_pixels=123,
        image_max_pixels=456,
    )
    _, _, trainer = build_unsloth_trainer(config)

    assert trainer.train_dataset[0]["messages"][1]["role"] == "assistant"
    assert calls["collator"] == {"max_seq_length": 1024, "completion_only_loss": True}
    processor = trainer.processing_class
    assert processor.image_processor.min_pixels == 123
    assert processor.image_processor.max_pixels == 456
    assert calls["peft"]["finetune_vision_layers"] is False


def _example(image_path: Path) -> Example:
    return Example(
        case_id="case-1",
        task_type="mcq",
        question="Which finding is present?",
        options=(("A", "Normal"), ("B", "Finding")),
        answer="B",
        image_path=image_path,
    )


def test_unsloth_record_uses_multimodal_response_conversation(tmp_path: Path) -> None:
    image_path = tmp_path / "case.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    record = to_unsloth_record(_example(image_path))
    messages = record["messages"]

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert [part["type"] for part in messages[0]["content"]] == ["text", "image"]
    assert messages[0]["content"][1]["image"].size == (8, 8)
    assert messages[1]["content"][0]["text"] == "B: Finding"


def test_generation_messages_omit_assistant_target(tmp_path: Path) -> None:
    example = _example(tmp_path / "case.png")

    messages = build_messages(example, image="case.png")

    assert len(messages) == 1
    assert messages[0]["content"][1] == {"type": "image", "image": "case.png"}


def test_image_references_cannot_escape_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        case_to_example(
            {
                "case_id": "case-1",
                "question type": "open-ended",
                "question": "Describe the image.",
                "image_path": "../outside.png",
            },
            tmp_path / "images",
            labeled=False,
        )


def test_unsloth_config_rejects_zero_trainable_modalities(tmp_path: Path) -> None:
    example = _example(tmp_path / "case.png")

    with pytest.raises(ValueError, match="at least one"):
        UnslothVLMConfig(
            model="test-vlm",
            output_dir=tmp_path / "out",
            train_examples=(example,),
            eval_examples=(example,),
            finetune_vision_layers=False,
            finetune_language_layers=False,
        )
