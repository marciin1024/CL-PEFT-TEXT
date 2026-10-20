"""Bottleneck adapters — custom implementation (plan's Option B).

The off-the-shelf `adapters` library pins `transformers~=4.45.2`, which
conflicts with the rest of our stack, so we implement the adapter ourselves —
a ~30-line module plus a hook-based injector. This keeps the second PEFT
family in the comparison without a dependency conflict.

A `BottleneckAdapter` is a down-projection -> GELU -> up-projection with a
residual connection, inserted after each transformer layer. The up-projection
is zero-initialized so the adapter starts as the identity and training is
stable from step one (standard Houlsby/AdapterHub practice).

`add_bottleneck_adapters` freezes the base, attaches one adapter per RoBERTa
layer, and registers forward hooks that apply it to the layer output. Returns
the encoder and the list of hook handles (for clean removal if needed).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, reduction_factor: int = 16) -> None:
        super().__init__()
        bottleneck = max(1, dim // reduction_factor)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        # Near-identity init: adapter contributes ~0 at the start.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


def _get_layers(encoder: PreTrainedModel) -> nn.ModuleList:
    """Return the transformer layer stack. RoBERTa/BERT expose it at
    `encoder.encoder.layer`."""
    base = encoder
    if not hasattr(base, "encoder") or not hasattr(base.encoder, "layer"):
        raise AttributeError(
            "Could not find `.encoder.layer` on the backbone. "
            "Bottleneck adapters are wired for RoBERTa/BERT-style models; "
            "adapt _get_layers for a different architecture."
        )
    return base.encoder.layer


def _make_hook(layer: nn.Module):
    """Forward hook that runs the layer's attached adapter on its output.

    RoBERTa layers return a tuple `(hidden_states, ...)`; we transform element
    0 and pass the rest through unchanged.
    """

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = module.bottleneck_adapter(output[0])
            return (hidden,) + output[1:]
        return module.bottleneck_adapter(output)

    return hook


def add_bottleneck_adapters(
    encoder: PreTrainedModel, reduction_factor: int = 16
) -> Tuple[PreTrainedModel, List[torch.utils.hooks.RemovableHandle]]:
    """Freeze the base and attach a trainable bottleneck adapter to each layer.

    Args:
        encoder: a frozen (or to-be-frozen) RoBERTa-style encoder.
        reduction_factor: bottleneck = hidden_size // reduction_factor.
            Larger factor -> smaller adapter (the Phase-6 bottleneck ablation
            sweeps this).
    """
    for p in encoder.parameters():
        p.requires_grad = False

    dim = encoder.config.hidden_size
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for layer in _get_layers(encoder):
        layer.add_module("bottleneck_adapter", BottleneckAdapter(dim, reduction_factor))
        handles.append(layer.register_forward_hook(_make_hook(layer)))
    return encoder, handles
