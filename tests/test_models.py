"""Unit tests for the Phase-2 model components (CPU, no weight download).

A tiny RoBERTa config (2 layers, hidden 32) is constructed from scratch via
`AutoModel.from_config`, so these tests run offline and fast. They verify:
  * CosineHead.expand grows rows and preserves existing ones
  * cosine logits have the right shape and bounded magnitude
  * LoRA wrapping leaves only LoRA + head trainable, base frozen
  * the custom bottleneck adapter path trains only adapters + head
  * "full" unfreezes the backbone; "linear" trains only the head

Run:  python -m tests.test_models      (or pytest tests/test_models.py)
"""

from __future__ import annotations

import torch
from transformers import AutoModel, RobertaConfig

from src.models.backbone import count_trainable_params
from src.models.build import build_model
from src.models.cosine_head import CosineHead

TINY = RobertaConfig(
    vocab_size=200,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=66,
)


def _tiny_encoder():
    return AutoModel.from_config(TINY)


def _fake_batch(bsz=4, seqlen=8):
    input_ids = torch.randint(0, TINY.vocab_size, (bsz, seqlen))
    attention_mask = torch.ones(bsz, seqlen, dtype=torch.long)
    return input_ids, attention_mask


# ---- CosineHead --------------------------------------------------------

def test_cosine_head_expand_preserves_rows():
    head = CosineHead(in_features=32, num_classes=5)
    before = head.weight.data.clone()
    head.expand(3)
    assert head.num_classes == 8
    # first 5 rows untouched
    assert torch.equal(head.weight.data[:5], before)


def test_cosine_head_forward_shape_and_scale():
    head = CosineHead(in_features=32, num_classes=7, scale=16.0)
    x = torch.randn(4, 32)
    logits = head(x)
    assert logits.shape == (4, 7)
    # cosine in [-1,1] * scale -> bounded by scale
    assert logits.abs().max().item() <= 16.0 + 1e-4


def test_cosine_head_empty_then_grow():
    head = CosineHead(in_features=32, num_classes=0)
    assert head.num_classes == 0
    head.expand(4)
    assert head(torch.randn(2, 32)).shape == (2, 4)


# ---- build_model param accounting (the Phase-2 DoD invariant) ----------

def _trainable_module_names(model):
    return {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}


def test_lora_trains_only_lora_and_head(monkeypatch):
    # Patch load_backbone to use the tiny offline encoder.
    import src.models.build as build_mod

    monkeypatch.setattr(build_mod, "load_backbone", lambda name, freeze=True: _set_freeze(_tiny_encoder(), freeze))
    model = build_model({"peft": "lora", "lora_r": 4}, "tiny", num_initial_classes=5)

    # base encoder weights must be frozen; only lora_* and head trainable
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("lora_" in n for n in trainable_names), "no LoRA params trainable"
    assert any(n.startswith("head.") for n in trainable_names), "head not trainable"
    assert not any(
        ("lora_" not in n and not n.startswith("head.")) for n in trainable_names
    ), f"unexpected trainable params: {trainable_names}"

    # forward works
    ids, mask = _fake_batch()
    assert model(ids, mask).shape == (4, 5)


def test_adapter_trains_only_adapters_and_head(monkeypatch):
    import src.models.build as build_mod

    monkeypatch.setattr(build_mod, "load_backbone", lambda name, freeze=True: _set_freeze(_tiny_encoder(), freeze))
    model = build_model({"peft": "adapter", "reduction_factor": 8}, "tiny", num_initial_classes=6)

    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(
        ("bottleneck_adapter" in n) or n.startswith("head.") for n in trainable_names
    ), f"unexpected trainable params: {trainable_names}"
    ids, mask = _fake_batch()
    assert model(ids, mask).shape == (4, 6)


def test_linear_probe_trains_only_head(monkeypatch):
    import src.models.build as build_mod

    monkeypatch.setattr(build_mod, "load_backbone", lambda name, freeze=True: _set_freeze(_tiny_encoder(), freeze))
    model = build_model({"peft": "linear"}, "tiny", num_initial_classes=3)
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith("head.") for n in trainable_names)


def test_full_ft_unfreezes_backbone(monkeypatch):
    import src.models.build as build_mod

    monkeypatch.setattr(build_mod, "load_backbone", lambda name, freeze=True: _set_freeze(_tiny_encoder(), freeze))
    model = build_model({"peft": "full"}, "tiny", num_initial_classes=3)
    trainable, total = count_trainable_params(model)
    # nearly everything trainable
    assert trainable > 0.9 * total


def _set_freeze(model, freeze):
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    return model


if __name__ == "__main__":
    # Minimal manual runner (no pytest / monkeypatch).
    test_cosine_head_expand_preserves_rows()
    test_cosine_head_forward_shape_and_scale()
    test_cosine_head_empty_then_grow()

    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    mp = _MP()
    test_lora_trains_only_lora_and_head(mp)
    test_adapter_trains_only_adapters_and_head(mp)
    test_linear_probe_trains_only_head(mp)
    test_full_ft_unfreezes_backbone(mp)
    print("All model tests passed.")
