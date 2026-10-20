"""Frozen text encoder, sentence pooling, and trainable-parameter accounting.

The backbone (roberta-base by default) is loaded with `requires_grad=False`
on every parameter — only PEFT modules and the cosine head ever receive
gradients. Pooling turns the token-level `last_hidden_state` into a single
sentence vector for the classifier head.
"""

from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoModel, PreTrainedModel


def load_backbone(name: str, freeze: bool = True) -> PreTrainedModel:
    """Load an encoder and (by default) freeze all of its parameters."""
    model = AutoModel.from_pretrained(name)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    return model


def hidden_size(model: PreTrainedModel) -> int:
    return model.config.hidden_size


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mask-aware mean pooling over the sequence dimension.

    Padding tokens (attention_mask == 0) are excluded from the average, so
    the dynamic padding used by the collator does not bias the sentence
    representation.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Take the first-token (<s> / [CLS]) representation."""
    return last_hidden_state[:, 0]


def count_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def trainable_summary(model: torch.nn.Module) -> str:
    trainable, total = count_trainable_params(model)
    pct = 100.0 * trainable / total if total else 0.0
    return f"trainable {trainable:,} / {total:,} params ({pct:.3f}%)"
