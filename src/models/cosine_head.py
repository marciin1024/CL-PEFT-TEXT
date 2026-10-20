"""Growable cosine classification head for class-incremental learning.

Logits are `s * cos(x, W_c)` — features and class weights are L2-normalized
and scaled by `s` (~16). The cosine geometry curbs the task-recency bias that
plagues plain linear CIL heads (later tasks' weights grow in norm and
dominate), and removes the need to maintain Nearest-Class-Mean centroids that
would drift as LoRA reshapes the feature space.

`expand(n)` appends `n` new class rows **without touching existing rows**, so
old classes keep their learned direction when a new task arrives.

IMPORTANT (optimizer): `expand` replaces the `weight` Parameter object. Any
optimizer created before the call still references the old tensor and will not
update the new rows. The continual trainer (Phase 3) therefore rebuilds the
optimizer after growing the head at each task boundary.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 0, scale: float = 16.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        if num_classes > 0:
            self._init_rows(self.weight)

    @staticmethod
    def _init_rows(tensor: torch.Tensor) -> None:
        # Kaiming-uniform like nn.Linear; rows are L2-normalized at forward
        # time so the exact scale of init matters little.
        if tensor.numel() > 0:
            nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

    @property
    def num_classes(self) -> int:
        return self.weight.shape[0]

    @torch.no_grad()
    def expand(self, num_new_classes: int) -> "CosineHead":
        """Append `num_new_classes` freshly-initialized rows, preserving the
        existing ones. Returns self for chaining."""
        if num_new_classes <= 0:
            return self
        device = self.weight.device
        dtype = self.weight.dtype
        new_rows = torch.empty(num_new_classes, self.in_features, device=device, dtype=dtype)
        self._init_rows(new_rows)
        combined = torch.cat([self.weight.data, new_rows], dim=0)
        self.weight = nn.Parameter(combined)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 0:
            raise RuntimeError("CosineHead has no classes yet; call expand() first.")
        x = F.normalize(x, p=2, dim=-1)
        w = F.normalize(self.weight, p=2, dim=-1)
        return self.scale * F.linear(x, w)
