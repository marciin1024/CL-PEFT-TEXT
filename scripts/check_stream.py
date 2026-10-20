"""Phase-1 DoD check.

Builds the Banking77 CIL stream from config/base.yaml, prints the summary,
runs hard validation (disjoint classes, no label leaks, global label range)
and shows that a different class_order_seed yields a different valid
partition. Finally pulls one batch from a train loader to prove the
tokenized pipeline works end-to-end.

Run from the repo root (needs network for the HF Hub on first run):
    python -m scripts.check_stream
    python -m scripts.check_stream --sanity        # AG News harness
"""

from __future__ import annotations

import argparse

import yaml

from src.data.cil_split import build_cil_split, describe_split, validate_split
from src.data.load import load_text_classification
from src.data.streams import build_task_stream
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 DoD check")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--sanity", action="store_true", help="use the AG News sanity stream")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    section = cfg["sanity"] if args.sanity else cfg["data"]
    name = section["dataset"]

    # --- raw split + validation -----------------------------------------
    data = load_text_classification(name, cfg["data"]["val_fraction"], cfg["seed"])
    split = build_cil_split(
        data["train"], data["val"], data["test"],
        num_classes=data["num_classes"],
        num_tasks=section["num_tasks"],
        class_order_seed=section["class_order_seed"],
    )
    print(describe_split(split))
    validate_split(split)
    print("validate_split: OK (disjoint classes, no leaks, full coverage)")

    # global label range check
    all_labels = set()
    for task in split.tasks:
        all_labels |= set(task.train["label"])
    lo, hi = min(all_labels), max(all_labels)
    assert lo >= 0 and hi == data["num_classes"] - 1
    print(f"Labels are global ids in [0, {data['num_classes'] - 1}]: OK")

    # --- different seed -> different valid partition ---------------------
    split2 = build_cil_split(
        data["train"], data["val"], data["test"],
        num_classes=data["num_classes"],
        num_tasks=section["num_tasks"],
        class_order_seed=section["class_order_seed"] + 1,
    )
    validate_split(split2)
    assert split.class_order != split2.class_order
    print("Second class_order_seed gives a different valid partition: OK")

    # --- tokenized loaders end-to-end ------------------------------------
    stream = build_task_stream(cfg, sanity=args.sanity)
    batch = next(iter(stream.train_loader(0)))
    print(
        "First batch from task 0: "
        f"input_ids={tuple(batch['input_ids'].shape)}, "
        f"attention_mask={tuple(batch['attention_mask'].shape)}, "
        f"labels={tuple(batch['labels'].shape) if 'labels' in batch else tuple(batch['label'].shape)}"
    )
    print("Phase-1 DoD: ALL GREEN")


if __name__ == "__main__":
    main()
