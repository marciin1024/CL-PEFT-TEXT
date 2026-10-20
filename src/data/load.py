"""Dataset loading for the CL benchmark.

Primary benchmark: Banking77 (77 customer-support intents, short utterances).
Sanity benchmark:  AG News (4 topic classes).

Banking77 on the Hub ships only train/test, so a validation split is carved
out of train with a stratified, seeded split. Everything downstream consumes
the uniform dict returned by `load_text_classification`:

    {"train": Dataset, "val": Dataset, "test": Dataset,
     "num_classes": int, "text_column": str}

with a guaranteed `label` column containing *global* integer class ids.
"""

from __future__ import annotations

from typing import Any, Dict

from datasets import Dataset, DatasetDict, load_dataset

# dataset name -> (hub id, text column)
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "banking77": {"hub_id": "banking77", "text_column": "text"},
    "ag_news": {"hub_id": "ag_news", "text_column": "text"},
}


def _carve_val(train: Dataset, val_fraction: float, seed: int) -> DatasetDict:
    """Stratified train/val split (stratify by label so every class keeps
    roughly the same train/val ratio — important when tasks later filter
    by class)."""
    split = train.train_test_split(
        test_size=val_fraction,
        seed=seed,
        stratify_by_column="label",
    )
    return DatasetDict(train=split["train"], val=split["test"])


def load_text_classification(
    name: str,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Load a registered dataset and return uniform train/val/test splits.

    Args:
        name: "banking77" or "ag_news".
        val_fraction: fraction of train carved into val when the dataset
            has no validation split of its own.
        seed: seed for the stratified val carve (independent of the
            class-order seed used by the CIL split).
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(_REGISTRY)}")
    spec = _REGISTRY[name]

    raw = load_dataset(spec["hub_id"])

    # `label` must be a ClassLabel column for stratify_by_column.
    train = raw["train"]
    test = raw["test"]

    if "validation" in raw:
        val = raw["validation"]
    else:
        carved = _carve_val(train, val_fraction, seed)
        train, val = carved["train"], carved["val"]

    num_classes = train.features["label"].num_classes

    return {
        "train": train,
        "val": val,
        "test": test,
        "num_classes": num_classes,
        "text_column": spec["text_column"],
    }
