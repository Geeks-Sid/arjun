"""Phase 02 smoke: the whole contract surface end to end on CPU.

Run directly via the phase smoke command:
    pytest tests/phase_02/test_contract_smoke.py -q
"""

from dataclasses import replace

import torch
from contract_fixtures import (
    DummyLanguageModelAdapter,
    DummyTaskModule,
    DummyVisualEncoder,
    make_batch,
    make_sample,
)

from medfm.core import (
    BucketId,
    BucketKind,
    GeneratedText,
    GenerationConfig,
    LanguageModelAdapter,
    MedicalSample,
    Modality,
    OutputSpec,
    ProjectedVisualTokens,
    TaskModule,
    VisualEncoder,
    canonical_json,
    config_hash,
)


def test_contract_smoke_samples_for_all_modality_families():
    for modality in Modality:
        sample = make_sample(modality)
        restored = MedicalSample.from_dict(sample.to_dict())
        assert restored.to_dict() == sample.to_dict()
        canonical_json(sample.to_dict())  # serializable deterministically


def test_contract_smoke_batches_for_all_modality_families():
    for modality in Modality:
        batch = make_batch(modality)
        assert batch.device is not None
        canonical_json(batch.to_metadata_dict())


def test_contract_smoke_encoder_lm_task_pipeline():
    encoder = DummyVisualEncoder()
    lm = DummyLanguageModelAdapter()
    task = DummyTaskModule()
    assert isinstance(encoder, VisualEncoder)
    assert isinstance(lm, LanguageModelAdapter)
    assert isinstance(task, TaskModule)

    batch = make_batch(Modality.XRAY_2D, batch_size=2)
    output = encoder.encode(batch)
    output.check_against(OutputSpec(pooled=True, spatial_tokens=True))

    visual = ProjectedVisualTokens(
        tokens=torch.randn(2, 4, 8),
        source_modality=Modality.XRAY_2D,
        token_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    text = lm.tokenize(["describe", "describe"])
    lm_output = lm.forward_with_visual_tokens(text, visual, labels=text.input_ids)
    assert lm_output.logits is not None
    generated = lm.generate(text, visual, generation_config=GenerationConfig(max_new_tokens=4))
    assert isinstance(generated, GeneratedText)

    labeled = replace(batch, labels=torch.tensor([0, 1]))
    task.reset_metrics()
    task.update_metrics(encoder.encode(labeled), labeled)
    assert 0.0 <= task.compute_metrics()["accuracy"] <= 1.0


def test_contract_smoke_bucket_and_hash():
    batch = make_batch(Modality.TEXT_ONLY)
    bucketed = type(batch)(
        modality=batch.modality,
        sample_ids=batch.sample_ids,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        bucket=BucketId(kind=BucketKind.TEXT_TOKENS, shape=(8,)),
    )
    assert bucketed.bucket is not None
    assert config_hash(bucketed.to_metadata_dict()) == config_hash(bucketed.to_metadata_dict())
