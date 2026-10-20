"""Class-incremental (CIL) task split.

Shuffles the C global class ids with a fixed `class_order_seed`, partitions
them into T disjoint tasks, and filters each split (train/val/test) by the
task's class set. Labels remain **global** ids in [0, C-1] throughout — heads
and losses must never see per-task re-indexed labels (Gotcha #1).

Also exposes, per task t:
  * per-task test sets        -> accuracy matrix R[i][j] (forgetting / BWT)
  * cumulative test sets      -> CIL headline accuracy (all classes seen so far)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from datasets import Dataset, concatenate_datasets


@dataclass
class CILTask:
    """One task in the class-incremental stream."""

    task_id: int
    class_ids: List[int]            # global class ids introduced by this task
    train: Dataset
    val: Dataset
    test: Dataset                   # per-task test set (only this task's classes)


@dataclass
class CILSplit:
    """The full class-incremental stream."""

    tasks: List[CILTask]
    class_order: List[int]          # the shuffled global class order
    num_classes: int

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def classes_seen(self, t: int) -> List[int]:
        """All global class ids introduced in tasks 0..t (inclusive)."""
        out: List[int] = []
        for task in self.tasks[: t + 1]:
            out.extend(task.class_ids)
        return out

    def cumulative_test(self, t: int) -> Dataset:
        """Test set over all classes seen after finishing task t —
        the CIL headline evaluation."""
        return concatenate_datasets([task.test for task in self.tasks[: t + 1]])


def _partition_classes(
    num_classes: int, num_tasks: int, class_order_seed: int
) -> List[List[int]]:
    """Shuffle class ids with a dedicated RNG and split into T groups.

    Requires num_classes % num_tasks == 0 (e.g. 77 = 7 x 11). A dedicated
    np.random.Generator is used so the partition depends only on
    `class_order_seed`, not on global RNG state.
    """
    if num_classes % num_tasks != 0:
        raise ValueError(
            f"num_classes={num_classes} not divisible by num_tasks={num_tasks}"
        )
    rng = np.random.default_rng(class_order_seed)
    order = rng.permutation(num_classes).tolist()
    per_task = num_classes // num_tasks
    return [order[i * per_task : (i + 1) * per_task] for i in range(num_tasks)]


def _filter_by_classes(ds: Dataset, class_ids: Sequence[int]) -> Dataset:
    class_set = set(class_ids)
    # Batched filter over the label column — fast (arrow-level) and avoids
    # Python-object overhead per example.
    return ds.filter(
        lambda batch: [lbl in class_set for lbl in batch["label"]],
        batched=True,
    )


def build_cil_split(
    train: Dataset,
    val: Dataset,
    test: Dataset,
    num_classes: int,
    num_tasks: int,
    class_order_seed: int = 0,
) -> CILSplit:
    """Build the deterministic class-incremental stream.

    Args:
        train/val/test: full datasets with a global `label` column.
        num_classes: total number of classes (77 for Banking77).
        num_tasks: T (7 for the primary 7x11 setting).
        class_order_seed: governs the class-order permutation only.
    """
    groups = _partition_classes(num_classes, num_tasks, class_order_seed)

    tasks: List[CILTask] = []
    for t, class_ids in enumerate(groups):
        tasks.append(
            CILTask(
                task_id=t,
                class_ids=sorted(class_ids),
                train=_filter_by_classes(train, class_ids),
                val=_filter_by_classes(val, class_ids),
                test=_filter_by_classes(test, class_ids),
            )
        )

    class_order = [c for g in groups for c in g]
    return CILSplit(tasks=tasks, class_order=class_order, num_classes=num_classes)


def _cap_per_class(ds: Dataset, max_n: Optional[int], seed: int) -> Dataset:
    """Keep at most `max_n` examples per class (deterministic shuffle first).
    `max_n=None` returns the dataset unchanged."""
    if max_n is None:
        return ds
    ds = ds.shuffle(seed=seed)
    from collections import defaultdict

    counts: dict = defaultdict(int)
    keep: List[int] = []
    for i, lbl in enumerate(ds["label"]):
        if counts[lbl] < max_n:
            keep.append(i)
            counts[lbl] += 1
    return ds.select(keep)


def cap_split_per_class(
    split: CILSplit,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 0,
) -> CILSplit:
    """Subsample every task's train/val/test to at most N examples per class.

    Used to make the AG News sanity stream (and quick dev runs) fast — full
    AG News has ~30k examples per class, far more than a harness check needs.
    Banking77 (~130/class) is left unchanged when caps exceed its size.
    """
    for task in split.tasks:
        task.train = _cap_per_class(task.train, max_train, seed)
        task.val = _cap_per_class(task.val, max_val, seed)
        task.test = _cap_per_class(task.test, max_test, seed)
    return split


def describe_split(split: CILSplit) -> str:
    """Human-readable summary used by the Phase-1 DoD check."""
    lines = [
        f"CIL stream: {split.num_tasks} tasks, {split.num_classes} classes total"
    ]
    for task in split.tasks:
        lines.append(
            f"  task {task.task_id}: classes={task.class_ids} "
            f"| train={len(task.train)} val={len(task.val)} test={len(task.test)}"
        )
    return "\n".join(lines)


def validate_split(split: CILSplit) -> None:
    """Hard checks for the Phase-1 DoD:
    disjoint class sets, full coverage, no label leaks across task test sets.
    """
    seen: set[int] = set()
    for task in split.tasks:
        cset = set(task.class_ids)
        if cset & seen:
            raise AssertionError(f"Task {task.task_id} overlaps earlier classes")
        seen |= cset
        for split_name in ("train", "val", "test"):
            labels = set(getattr(task, split_name)["label"])
            if not labels <= cset:
                raise AssertionError(
                    f"Label leak in task {task.task_id} {split_name}: "
                    f"{labels - cset}"
                )
    if seen != set(range(split.num_classes)):
        raise AssertionError("Class partition does not cover all classes")
