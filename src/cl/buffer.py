"""Reservoir experience-replay buffer.

Stores **tokenized** samples (input_ids, attention_mask implied by length, and
the global label) — never cached embeddings, which would go stale the moment
LoRA updates the encoder's forward pass (proposal's key implementation
decision / Gotcha: "Replay = tokenized ids, never embeddings").

Reservoir sampling keeps a uniform random sample of the whole stream seen so
far within a fixed `capacity`, so the buffer never grows unbounded and early
tasks stay represented. Sequences are stored trimmed to their true length and
re-padded on `sample()`, so no padding memory is wasted.
"""

from __future__ import annotations

import random
from typing import Dict, List

import torch


class ReservoirBuffer:
    def __init__(self, capacity: int, pad_token_id: int = 1, seed: int = 42) -> None:
        self.capacity = capacity
        self.pad_token_id = pad_token_id
        self.rng = random.Random(seed)
        self.input_ids: List[torch.Tensor] = []  # list of 1-D LongTensors
        self.labels: List[int] = []               # global class ids
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self.labels)

    def is_empty(self) -> bool:
        return len(self.labels) == 0

    def _put(self, idx: int, ids: torch.Tensor, label: int) -> None:
        if idx == len(self.input_ids):
            self.input_ids.append(ids)
            self.labels.append(label)
        else:
            self.input_ids[idx] = ids
            self.labels[idx] = label

    def add_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Add a (padded) batch of tokenized samples via reservoir sampling.

        Each example is trimmed to its true length using its attention mask
        before storage, and moved to CPU so the buffer never pins VRAM.
        """
        bsz = input_ids.size(0)
        for i in range(bsz):
            length = int(attention_mask[i].sum().item())
            ids = input_ids[i, :length].detach().to("cpu").clone()
            label = int(labels[i].item())
            self.n_seen += 1
            if len(self.labels) < self.capacity:
                self._put(len(self.labels), ids, label)
            else:
                j = self.rng.randint(0, self.n_seen - 1)
                if j < self.capacity:
                    self._put(j, ids, label)

    def sample(self, n: int) -> Dict[str, torch.Tensor]:
        """Return a padded replay minibatch of up to `n` samples.

        Output keys: input_ids, attention_mask, labels (global ids). Tensors
        live on CPU; the trainer moves them to the device.
        """
        n = min(n, len(self.labels))
        idxs = self.rng.sample(range(len(self.labels)), n)
        seqs = [self.input_ids[i] for i in idxs]
        labels = torch.tensor([self.labels[i] for i in idxs], dtype=torch.long)

        max_len = max(s.size(0) for s in seqs)
        input_ids = torch.full((n, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        for k, s in enumerate(seqs):
            input_ids[k, : s.size(0)] = s
            attention_mask[k, : s.size(0)] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def class_distribution(self) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for lbl in self.labels:
            dist[lbl] = dist.get(lbl, 0) + 1
        return dist
