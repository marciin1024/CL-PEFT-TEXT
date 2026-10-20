"""Continual trainer.

Drives the class-incremental loop and produces the artifacts every metric in
Phase 5 needs:

  * accuracy matrix R[i][j] = accuracy on task j's test set AFTER finishing
    task i, evaluated in the CIL setting (argmax over ALL classes seen so far,
    no task-id);
  * cumulative all-classes accuracy after each task (the CIL headline);
  * per-task wall-clock training time and peak VRAM.

Per task t the trainer:
  1. registers task t's new global classes and grows the cosine head;
  2. rebuilds the optimizer (the head's Parameter object was replaced by
     `expand`, so a stale optimizer would not train the new rows — see
     CosineHead docstring);
  3. trains for `epochs`, optionally mixing an equal-size replay minibatch
     drawn from past tasks;
  4. adds task t's training data to the buffer (reservoir);
  5. evaluates on every task seen so far to fill row t of R.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from src.cl.buffer import ReservoirBuffer
from src.cl.label_map import LabelMapper
from src.utils.device import peak_vram_mb, reset_peak_vram


@dataclass
class CLResults:
    R: List[List[Optional[float]]]          # R[i][j], lower-triangular filled
    cumulative_acc: List[float]             # CIL headline after each task
    time_per_task: List[float]              # seconds
    peak_vram_per_task: List[float]         # MB
    num_classes_per_task: List[int]         # classes seen after each task
    extra: dict = field(default_factory=dict)


def _get_labels(batch) -> torch.Tensor:
    # collator may emit "labels" or "label"
    return batch["labels"] if "labels" in batch else batch["label"]


class ContinualTrainer:
    def __init__(
        self,
        model,
        stream,
        device: torch.device,
        epochs: int = 3,
        lr: float = 1e-3,
        autocast_dtype: Optional[torch.dtype] = None,
        buffer: Optional[ReservoirBuffer] = None,
        replay_ratio: float = 1.0,
        oracle_task_id: bool = False,
        optimizer_factory: Optional[Callable] = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.model = model
        self.stream = stream
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.autocast_dtype = autocast_dtype
        self.buffer = buffer
        self.replay_ratio = replay_ratio
        self.oracle_task_id = oracle_task_id
        self.log = log
        self.mapper = LabelMapper()
        self._opt_factory = optimizer_factory or (
            lambda params: torch.optim.AdamW(params, lr=self.lr)
        )

    # ---- helpers --------------------------------------------------------

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        )

    def _move(self, batch):
        return (
            batch["input_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
            _get_labels(batch).to(self.device),
        )

    @torch.no_grad()
    def _eval_loader(self, loader, restrict_to: Optional[List[int]] = None) -> float:
        """CIL accuracy over a loader. If `restrict_to` (head indices) is given
        — used only by the Task-IL oracle — argmax is restricted to those
        columns."""
        self.model.eval()
        correct = total = 0
        for batch in loader:
            ids, mask, labels = self._move(batch)
            with self._autocast():
                logits = self.model(ids, mask)
            if restrict_to is not None:
                mask_cols = torch.full_like(logits, float("-inf"))
                mask_cols[:, restrict_to] = logits[:, restrict_to]
                logits = mask_cols
            pred = logits.argmax(dim=-1)
            correct += (pred == self.mapper.to_idx(labels)).sum().item()
            total += labels.numel()
        return correct / max(total, 1)

    # ---- main loop ------------------------------------------------------

    def run(self) -> CLResults:
        T = self.stream.num_tasks
        R: List[List[Optional[float]]] = [[None] * T for _ in range(T)]
        cumulative_acc: List[float] = []
        time_per_task: List[float] = []
        peak_vram_per_task: List[float] = []
        classes_after: List[int] = []

        scaler = torch.amp.GradScaler(
            self.device.type, enabled=(self.autocast_dtype == torch.float16)
        )
        criterion = nn.CrossEntropyLoss()

        for t in range(T):
            # 1. grow head for this task's new classes
            new = self.mapper.register(sorted(self.stream.new_classes(t)))
            self.model.expand_head(new)
            self.model.to(self.device)

            # 2. fresh optimizer over the (now larger) trainable set
            optimizer = self._opt_factory(self.model.trainable_parameters())

            # 3. train
            reset_peak_vram()
            start = time.time()
            train_loader = self.stream.train_loader(t)
            for epoch in range(self.epochs):
                self.model.train()
                running = 0.0
                for batch in train_loader:
                    ids, mask, labels = self._move(batch)

                    # replay from PAST tasks (buffer holds tasks 0..t-1 here)
                    if self.buffer is not None and not self.buffer.is_empty():
                        n_replay = max(1, int(ids.size(0) * self.replay_ratio))
                        rep = self.buffer.sample(n_replay)
                        r_ids = rep["input_ids"].to(self.device)
                        r_mask = rep["attention_mask"].to(self.device)
                        r_labels = rep["labels"].to(self.device)
                        # pad the two batches to a common length before concat
                        ids, mask = _pad_concat(ids, mask, r_ids, r_mask, self.model)
                        labels = torch.cat([labels, r_labels], dim=0)

                    idx_labels = self.mapper.to_idx(labels)
                    optimizer.zero_grad()
                    with self._autocast():
                        logits = self.model(ids, mask)
                        loss = criterion(logits, idx_labels)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    running += loss.item()
                self.log(
                    f"  task {t} epoch {epoch+1}/{self.epochs} "
                    f"loss={running/max(len(train_loader),1):.4f}"
                )

            elapsed = time.time() - start
            peak = peak_vram_mb()

            # 4. add task t's data to the buffer (after training -> replay used
            #    only past tasks during this task)
            if self.buffer is not None:
                for batch in self.stream.train_loader(t):
                    self.buffer.add_batch(
                        batch["input_ids"], batch["attention_mask"], _get_labels(batch)
                    )

            # 5. evaluate on all tasks seen so far -> fill row t of R
            for j in range(t + 1):
                restrict = None
                if self.oracle_task_id:
                    restrict = [self.mapper.global_to_idx[c] for c in self.stream.new_classes(j)]
                R[t][j] = self._eval_loader(self.stream.test_loader(j), restrict_to=restrict)

            cum = self._eval_loader(self.stream.cumulative_test_loader(t))
            cumulative_acc.append(cum)
            time_per_task.append(elapsed)
            peak_vram_per_task.append(peak)
            classes_after.append(len(self.mapper))

            seen_accs = " ".join(f"{R[t][j]:.3f}" for j in range(t + 1))
            self.log(
                f"task {t} done: cum_acc={cum:.4f} | R[{t}][0..{t}]=[{seen_accs}] "
                f"| {elapsed:.1f}s | {peak:.0f}MB"
            )

        return CLResults(
            R=R,
            cumulative_acc=cumulative_acc,
            time_per_task=time_per_task,
            peak_vram_per_task=peak_vram_per_task,
            num_classes_per_task=classes_after,
        )


def _pad_concat(ids_a, mask_a, ids_b, mask_b, model):
    """Concatenate two (B, L) batches that may differ in sequence length,
    right-padding the shorter one with the head's pad id (0 is safe since the
    attention mask zeroes it out)."""
    la, lb = ids_a.size(1), ids_b.size(1)
    if la != lb:
        target = max(la, lb)
        ids_a, mask_a = _pad_to(ids_a, mask_a, target)
        ids_b, mask_b = _pad_to(ids_b, mask_b, target)
    return torch.cat([ids_a, ids_b], dim=0), torch.cat([mask_a, mask_b], dim=0)


def _pad_to(ids, mask, target):
    pad = target - ids.size(1)
    if pad <= 0:
        return ids, mask
    ids = torch.nn.functional.pad(ids, (0, pad), value=0)
    mask = torch.nn.functional.pad(mask, (0, pad), value=0)
    return ids, mask
