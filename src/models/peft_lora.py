"""LoRA injection on the frozen encoder via HuggingFace `peft`.

We wrap the *bare* encoder (AutoModel) rather than an
`AutoModelForSequenceClassification`, because classification is handled by our
own growable `CosineHead`. `TaskType.FEATURE_EXTRACTION` tells peft to inject
LoRA into the target modules and keep the base frozen, without managing any
classifier head of its own.

`target_modules=["query", "value"]` matches RoBERTa's attention projections
(`...attention.self.query/value`). DeBERTa-v3 uses different names
(`in_proj`/`pos_*`), so override `target_modules` from config if you switch
the backbone fallback.
"""

from __future__ import annotations

from typing import Sequence

from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel


def wrap_lora(
    encoder: PreTrainedModel,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Sequence[str] = ("query", "value"),
) -> PreTrainedModel:
    """Return `encoder` wrapped with a single LoRA adapter.

    The base weights stay frozen; only the injected low-rank A/B matrices are
    trainable. Rank `r`, `lora_alpha` and the target modules are exposed so the
    Phase-6 rank ablation can sweep them from config.
    """
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(encoder, config)
