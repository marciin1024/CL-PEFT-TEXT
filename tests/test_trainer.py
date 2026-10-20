"""Continual-trainer integration test on a tiny model + synthetic stream.

No network / GPU. A fake stream presents 3 tasks x 2 classes of synthetic
tokenized data; we verify the trainer fills a lower-triangular accuracy
matrix, grows the head across tasks, and runs both with and without replay.
"""

from __future__ import annotations

import torch
from datasets import ClassLabel, Dataset, Features, Value
from torch.utils.data import DataLoader
from transformers import AutoModel, DataCollatorWithPadding, RobertaConfig, RobertaTokenizerFast

import src.models.build as build_mod
from src.cl.buffer import ReservoirBuffer
from src.cl.trainer import ContinualTrainer
from src.models.build import build_model

TINY = RobertaConfig(
    vocab_size=50,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=40,
    pad_token_id=1,
)

NUM_CLASSES = 6
NUM_TASKS = 3
PER_TASK = NUM_CLASSES // NUM_TASKS


def _fake_load(name, freeze=True):
    m = AutoModel.from_config(TINY)
    if freeze:
        for p in m.parameters():
            p.requires_grad = False
    return m


build_mod.load_backbone = _fake_load


class _FakeStream:
    """Minimal TaskStream stand-in over synthetic tokenized tensors."""

    def __init__(self, n_per_class=24, seqlen=8):
        self.num_tasks = NUM_TASKS
        self.num_classes = NUM_CLASSES
        torch.manual_seed(0)
        # class-conditional token bias so the task is learnable above chance
        self._train, self._test = {}, {}
        for c in range(NUM_CLASSES):
            ids_tr = torch.randint(3 + c, 6 + c, (n_per_class, seqlen))
            ids_te = torch.randint(3 + c, 6 + c, (n_per_class, seqlen))
            self._train[c] = ids_tr
            self._test[c] = ids_te
        self.collator = self._collate

    def new_classes(self, t):
        return list(range(t * PER_TASK, (t + 1) * PER_TASK))

    def _make_ds(self, source, classes):
        ids = torch.cat([source[c] for c in classes], dim=0)
        labels = torch.cat(
            [torch.full((source[c].size(0),), c, dtype=torch.long) for c in classes]
        )
        return ids, labels

    def _loader(self, ids, labels, bs=16):
        data = [
            {"input_ids": ids[i], "attention_mask": torch.ones_like(ids[i]), "label": int(labels[i])}
            for i in range(ids.size(0))
        ]
        return DataLoader(data, batch_size=bs, shuffle=True, collate_fn=self._collate)

    @staticmethod
    def _collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def train_loader(self, t):
        ids, labels = self._make_ds(self._train, self.new_classes(t))
        return self._loader(ids, labels)

    def test_loader(self, t):
        ids, labels = self._make_ds(self._test, self.new_classes(t))
        return self._loader(ids, labels)

    def cumulative_test_loader(self, t):
        classes = [c for tt in range(t + 1) for c in self.new_classes(tt)]
        ids, labels = self._make_ds(self._test, classes)
        return self._loader(ids, labels)


def _run(replay: bool):
    model = build_model({"peft": "lora", "lora_r": 4}, "tiny", num_initial_classes=0)
    stream = _FakeStream()
    buffer = ReservoirBuffer(capacity=64, pad_token_id=1) if replay else None
    trainer = ContinualTrainer(
        model=model,
        stream=stream,
        device=torch.device("cpu"),
        epochs=2,
        lr=5e-2,
        autocast_dtype=None,
        buffer=buffer,
        log=lambda *_: None,
    )
    return trainer.run()


def test_R_matrix_is_lower_triangular_and_filled():
    res = _run(replay=False)
    R = res.R
    for i in range(NUM_TASKS):
        for j in range(NUM_TASKS):
            if j <= i:
                assert R[i][j] is not None, f"R[{i}][{j}] should be filled"
                assert 0.0 <= R[i][j] <= 1.0
            else:
                assert R[i][j] is None, f"R[{i}][{j}] should be empty (future task)"


def test_head_grows_across_tasks():
    res = _run(replay=False)
    assert res.num_classes_per_task == [2, 4, 6]
    assert len(res.cumulative_acc) == NUM_TASKS


def test_replay_runs_end_to_end():
    res = _run(replay=True)
    # all diagonal entries present and valid; just a smoke check that replay
    # path (sample + pad_concat + combined loss) executes
    assert all(res.R[i][i] is not None for i in range(NUM_TASKS))


if __name__ == "__main__":
    test_R_matrix_is_lower_triangular_and_filled()
    test_head_grows_across_tasks()
    test_replay_runs_end_to_end()
    print("All trainer tests passed.")
