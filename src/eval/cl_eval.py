"""Shared class-incremental evaluation.

`evaluate_cil` computes accuracy of a model over a loader in the CIL setting:
argmax over ALL head columns by default (no task-id). `restrict_to` (a list of
head-row indices) limits the argmax to a subset — used only by the Task-IL
oracle reference, where the task-id is honestly assumed known.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.cl.label_map import LabelMapper


@torch.no_grad()
def evaluate_cil(
    model,
    loader,
    mapper: LabelMapper,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    restrict_to: Optional[List[int]] = None,
) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
        ):
            logits = model(ids, mask)
        if restrict_to is not None:
            masked = torch.full_like(logits, float("-inf"))
            masked[:, restrict_to] = logits[:, restrict_to]
            logits = masked
        pred = logits.argmax(dim=-1)
        correct += (pred == mapper.to_idx(labels)).sum().item()
        total += labels.numel()
    return correct / max(total, 1)
