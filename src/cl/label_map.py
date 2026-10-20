"""Global-label <-> head-row mapping for class-incremental learning.

Banking77 labels are arbitrary global ids in [0, 76], but the cosine head is
indexed 0..K-1 and grows as classes arrive. The mapper assigns each new global
class the next free head row **in order of first appearance**, and that
assignment is permanent — a class seen in task 0 keeps its row forever, which
is exactly what lets old classes retain their learned direction as the head
grows (Gotcha #1: global labels, never reset per task).
"""

from __future__ import annotations

from typing import Iterable, List

import torch


class LabelMapper:
    def __init__(self) -> None:
        self.global_to_idx: dict[int, int] = {}
        self.idx_to_global: List[int] = []

    def __len__(self) -> int:
        return len(self.idx_to_global)

    def register(self, class_ids: Iterable[int]) -> int:
        """Register (possibly new) global class ids. Returns how many were new
        — i.e. how many rows the head must grow by."""
        new = 0
        for g in class_ids:
            g = int(g)
            if g not in self.global_to_idx:
                self.global_to_idx[g] = len(self.idx_to_global)
                self.idx_to_global.append(g)
                new += 1
        return new

    def to_idx(self, labels: torch.Tensor) -> torch.Tensor:
        """Map a tensor of global labels to head-row indices (same device)."""
        out = [self.global_to_idx[int(x)] for x in labels]
        return torch.tensor(out, dtype=torch.long, device=labels.device)
