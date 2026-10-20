"""Tokenization for the task stream.

Tokenizes with the backbone tokenizer (truncation at max_length=64 by
default) and pads *dynamically per batch* via DataCollatorWithPadding —
short Banking77 utterances make static padding wasteful.

Output examples carry exactly: input_ids, attention_mask, label
(label = global class id). The same tokenized form is what the replay
buffer will store in Phase 3 (never embeddings — they go stale under LoRA).
"""

from __future__ import annotations

from typing import Any, Dict

from datasets import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


def tokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    max_length: int = 64,
) -> Dataset:
    """Tokenize a dataset and keep only model inputs + the global label.

    Truncation is applied here; padding is deferred to the collator.
    """

    def _tok(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    drop = [c for c in ds.column_names if c != "label"]
    tokenized = ds.map(_tok, batched=True, remove_columns=drop)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


def make_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    """Dynamic-padding collator (pads each batch to its own longest seq)."""
    return DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
