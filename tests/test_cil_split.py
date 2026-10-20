"""Unit tests for src/data/cil_split.py on synthetic data (no network).

Builds a fake 77-class dataset shaped like Banking77 and checks the
Phase-1 DoD properties: 7 disjoint tasks of 11 classes, correct sizes,
global labels, no leaks, seed-dependent but deterministic partitions.

Run:  python -m pytest tests/test_cil_split.py -q
  or: python -m tests.test_cil_split
"""

from __future__ import annotations

import numpy as np
from datasets import ClassLabel, Dataset, Features, Value

from src.data.cil_split import build_cil_split, describe_split, validate_split

NUM_CLASSES = 77
NUM_TASKS = 7


def _fake_dataset(per_class: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(NUM_CLASSES), per_class)
    rng.shuffle(labels)
    texts = [f"utterance {i} for class {l}" for i, l in enumerate(labels)]
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(num_classes=NUM_CLASSES),
        }
    )
    return Dataset.from_dict(
        {"text": texts, "label": labels.tolist()}, features=features
    )


def _build(seed_order: int = 0):
    train = _fake_dataset(per_class=20, seed=1)
    val = _fake_dataset(per_class=4, seed=2)
    test = _fake_dataset(per_class=8, seed=3)
    return build_cil_split(
        train, val, test,
        num_classes=NUM_CLASSES,
        num_tasks=NUM_TASKS,
        class_order_seed=seed_order,
    )


def test_partition_shape():
    split = _build()
    assert split.num_tasks == NUM_TASKS
    for task in split.tasks:
        assert len(task.class_ids) == NUM_CLASSES // NUM_TASKS
        # 11 classes x 20/4/8 examples per class
        assert len(task.train) == 11 * 20
        assert len(task.val) == 11 * 4
        assert len(task.test) == 11 * 8


def test_validation_passes():
    split = _build()
    validate_split(split)  # raises on any violation


def test_global_labels():
    split = _build()
    all_labels = set()
    for task in split.tasks:
        all_labels |= set(task.train["label"])
    assert all_labels == set(range(NUM_CLASSES))


def test_cumulative_and_seen():
    split = _build()
    assert len(split.classes_seen(0)) == 11
    assert len(split.classes_seen(6)) == 77
    assert len(split.cumulative_test(2)) == 3 * 11 * 8


def test_seed_changes_partition_deterministically():
    a1, a2, b = _build(0), _build(0), _build(1)
    assert a1.class_order == a2.class_order      # deterministic
    assert a1.class_order != b.class_order       # seed-dependent
    validate_split(b)                            # still valid


if __name__ == "__main__":
    test_partition_shape()
    test_validation_passes()
    test_global_labels()
    test_cumulative_and_seen()
    test_seed_changes_partition_deterministically()
    print(describe_split(_build()).splitlines()[0])
    print("All cil_split tests passed.")
