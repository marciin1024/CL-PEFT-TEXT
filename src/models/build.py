"""Model factory.

`build_model(method_cfg, backbone_name)` returns a single `CLModel` that
bundles the (PEFT-wrapped) encoder, the pooling function and the growable
cosine head, exposing one `forward(input_ids, attention_mask) -> logits` and
an `expand_head(n)` for task boundaries.

The `method_cfg["peft"]` switch selects the adaptation strategy and is the
single knob every Phase-4 method config flips:

    "lora"   -> LoRA on attention query/value (frozen base)
    "adapter"-> custom bottleneck adapters     (frozen base)
    "linear" -> frozen base, only the head trains  (linear-probe reference)
    "full"   -> unfreeze the whole backbone        (naive FT lower bound)

⚠ Do not request both LoRA and bottleneck adapters on one instance — they are
separate method code paths to avoid hook/peft conflicts (plan's warning).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from src.models.backbone import (
    cls_pool,
    count_trainable_params,
    hidden_size,
    load_backbone,
    mean_pool,
)
from src.models.cosine_head import CosineHead
from src.models.peft_adapter import add_bottleneck_adapters
from src.models.peft_lora import wrap_lora


def _pooler(name: str) -> Callable[..., torch.Tensor]:
    if name == "mean":
        return lambda out, mask: mean_pool(out.last_hidden_state, mask)
    if name == "cls":
        return lambda out, mask: cls_pool(out.last_hidden_state)
    raise ValueError(f"Unknown pooling '{name}' (use 'mean' or 'cls')")


class CLModel(nn.Module):
    """Encoder (+PEFT) + pooling + growable cosine head."""

    def __init__(
        self,
        encoder: PreTrainedModel,
        head: CosineHead,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pooling_name = pooling
        self._pool = _pooler(pooling)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(input_ids, attention_mask))

    def expand_head(self, num_new_classes: int) -> None:
        self.head.expand(num_new_classes)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def param_counts(self):
        return count_trainable_params(self)


def build_model(
    method_cfg: Dict[str, Any],
    backbone_name: str,
    num_initial_classes: int = 0,
    pooling: str = "mean",
    scale: float = 16.0,
) -> CLModel:
    """Construct a CLModel for the given method configuration.

    Args:
        method_cfg: dict with at least `peft` in {lora, adapter, linear, full}
            and any method-specific hyperparameters (LoRA r/alpha/dropout,
            adapter reduction_factor).
        backbone_name: HF id, e.g. "roberta-base".
        num_initial_classes: rows to pre-allocate on the head (0 = grow later).
        pooling: "mean" (mask-aware) or "cls".
        scale: cosine-head logit scale s.
    """
    peft = method_cfg.get("peft", "linear")

    # "full" trains the backbone, so it must NOT be frozen at load time.
    encoder = load_backbone(backbone_name, freeze=(peft != "full"))

    if peft == "lora":
        encoder = wrap_lora(
            encoder,
            r=method_cfg.get("lora_r", 8),
            lora_alpha=method_cfg.get("lora_alpha", 16),
            lora_dropout=method_cfg.get("lora_dropout", 0.1),
            target_modules=method_cfg.get("target_modules", ("query", "value")),
        )
        dim = encoder.base_model.model.config.hidden_size
    elif peft == "adapter":
        encoder, _ = add_bottleneck_adapters(
            encoder, reduction_factor=method_cfg.get("reduction_factor", 16)
        )
        dim = hidden_size(encoder)
    elif peft in ("linear", "full"):
        dim = hidden_size(encoder)
    else:
        raise ValueError(f"Unknown peft mode '{peft}'")

    head = CosineHead(in_features=dim, num_classes=num_initial_classes, scale=scale)
    return CLModel(encoder=encoder, head=head, pooling=pooling)
