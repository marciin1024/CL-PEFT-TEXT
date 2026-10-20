"""TaskStream — the single object the continual trainer consumes.

Per task t it exposes:
    train_loader(t), val_loader(t), test_loader(t),
    cumulative_test_loader(t)   # all classes seen so far (CIL headline)
    classes_seen(t)             # global class ids introduced up to task t
    new_classes(t)              # classes introduced *by* task t (head growth)

Everything is parametric from config: T, class_order_seed, max_length,
batch sizes, num_workers. Shuffling uses a seeded torch.Generator so runs
are reproducible end-to-end.
"""

from __future__ import annotations

from typing import Any, Dict, List

from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.cil_split import CILSplit, build_cil_split, cap_split_per_class
from src.data.load import load_text_classification
from src.data.tokenize import make_collator, tokenize_dataset
from src.utils.seed import make_generator, seed_worker


class TaskStream:
    """Class-incremental stream of tokenized DataLoaders."""

    def __init__(
        self,
        split: CILSplit,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str,
        max_length: int = 64,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 2,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.tokenizer = tokenizer
        self.collator = make_collator(tokenizer)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Tokenize each task's splits once, up-front.
        self._train: List[Any] = []
        self._val: List[Any] = []
        self._test: List[Any] = []
        for task in split.tasks:
            self._train.append(
                tokenize_dataset(task.train, tokenizer, text_column, max_length)
            )
            self._val.append(
                tokenize_dataset(task.val, tokenizer, text_column, max_length)
            )
            self._test.append(
                tokenize_dataset(task.test, tokenizer, text_column, max_length)
            )

    # ----- properties ----------------------------------------------------

    @property
    def num_tasks(self) -> int:
        return self.split.num_tasks

    @property
    def num_classes(self) -> int:
        return self.split.num_classes

    def classes_seen(self, t: int) -> List[int]:
        return self.split.classes_seen(t)

    def new_classes(self, t: int) -> List[int]:
        return self.split.tasks[t].class_ids

    # ----- loaders --------------------------------------------------------

    def _loader(self, ds, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=make_generator(self.seed) if shuffle else None,
            pin_memory=True,
        )

    def train_loader(self, t: int) -> DataLoader:
        return self._loader(self._train[t], shuffle=True, batch_size=self.batch_size)

    def val_loader(self, t: int) -> DataLoader:
        return self._loader(self._val[t], shuffle=False, batch_size=self.eval_batch_size)

    def test_loader(self, t: int) -> DataLoader:
        """Per-task test loader — feeds the accuracy matrix R[i][j]."""
        return self._loader(self._test[t], shuffle=False, batch_size=self.eval_batch_size)

    def cumulative_test_loader(self, t: int) -> DataLoader:
        """All-classes-seen-so-far test loader — the CIL headline metric."""
        cumulative = concatenate_datasets(self._test[: t + 1])
        return self._loader(cumulative, shuffle=False, batch_size=self.eval_batch_size)

    def joint_train_loader(self) -> DataLoader:
        """Pooled train set over all tasks — for the joint upper bound (Phase 4)."""
        pooled = concatenate_datasets(self._train)
        return self._loader(pooled, shuffle=True, batch_size=self.batch_size)


def build_task_stream(cfg: Dict[str, Any], sanity: bool = False) -> TaskStream:
    """Factory: config dict -> ready TaskStream.

    Args:
        cfg: parsed base.yaml (expects `backbone` and `data`/`sanity` keys).
        sanity: if True, build the AG News sanity stream instead of Banking77.
    """
    section = cfg["sanity"] if sanity else cfg["data"]
    dataset_name = section["dataset"]

    data = load_text_classification(
        dataset_name,
        val_fraction=cfg["data"].get("val_fraction", 0.1),
        seed=cfg.get("seed", 42),
    )

    split = build_cil_split(
        train=data["train"],
        val=data["val"],
        test=data["test"],
        num_classes=data["num_classes"],
        num_tasks=section["num_tasks"],
        class_order_seed=section.get("class_order_seed", 0),
    )

    # Optional per-class caps (keeps the sanity stream / quick dev runs fast).
    max_train = section.get("max_train_per_class")
    max_val = section.get("max_val_per_class")
    max_test = section.get("max_test_per_class")
    if any(x is not None for x in (max_train, max_val, max_test)):
        split = cap_split_per_class(
            split, max_train, max_val, max_test, seed=cfg.get("seed", 42)
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"]["name"])

    return TaskStream(
        split=split,
        tokenizer=tokenizer,
        text_column=data["text_column"],
        max_length=cfg["backbone"].get("max_length", 64),
        batch_size=cfg["data"].get("batch_size", 32),
        eval_batch_size=cfg["data"].get("eval_batch_size", 64),
        num_workers=cfg["data"].get("num_workers", 2),
        seed=cfg.get("seed", 42),
    )
