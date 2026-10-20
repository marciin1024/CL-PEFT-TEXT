"""Phase-4 runner test: every method in the registry runs end-to-end on a tiny
synthetic stream and produces the strategy-appropriate R (CPU, no network)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, RobertaConfig

import src.models.build as build_mod
from src.cl.methods import all_methods
from src.cl.runner import run_method

TINY = RobertaConfig(
    vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
    intermediate_size=32, max_position_embeddings=40, pad_token_id=1,
)
NUM_CLASSES, NUM_TASKS = 6, 3
PER_TASK = NUM_CLASSES // NUM_TASKS


def _fake_load(name, freeze=True):
    m = AutoModel.from_config(TINY)
    if freeze:
        for p in m.parameters():
            p.requires_grad = False
    return m


build_mod.load_backbone = _fake_load


class _Tok:
    pad_token_id = 1


class _FakeStream:
    def __init__(self, n_per_class=16, seqlen=8):
        self.num_tasks = NUM_TASKS
        self.num_classes = NUM_CLASSES
        self.tokenizer = _Tok()
        torch.manual_seed(0)
        self._train, self._test = {}, {}
        for c in range(NUM_CLASSES):
            self._train[c] = torch.randint(3 + c, 6 + c, (n_per_class, seqlen))
            self._test[c] = torch.randint(3 + c, 6 + c, (n_per_class, seqlen))

    def new_classes(self, t):
        return list(range(t * PER_TASK, (t + 1) * PER_TASK))

    @staticmethod
    def _collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.tensor([b["label"] for b in batch]),
        }

    def _loader(self, source, classes, bs=16):
        data = []
        for c in classes:
            for i in range(source[c].size(0)):
                data.append({
                    "input_ids": source[c][i],
                    "attention_mask": torch.ones_like(source[c][i]),
                    "label": c,
                })
        return DataLoader(data, batch_size=bs, shuffle=True, collate_fn=self._collate)

    def train_loader(self, t):
        return self._loader(self._train, self.new_classes(t))

    def test_loader(self, t):
        return self._loader(self._test, self.new_classes(t))

    def cumulative_test_loader(self, t):
        classes = [c for tt in range(t + 1) for c in self.new_classes(tt)]
        return self._loader(self._test, classes)

    def joint_train_loader(self):
        return self._loader(self._train, list(range(NUM_CLASSES)))


def _run(name):
    return run_method(
        name, {"backbone": {"name": "tiny"}, "seed": 0}, _FakeStream(),
        torch.device("cpu"), epochs=1, lr=5e-2, autocast_dtype=None, log=lambda *_: None,
    )


def test_all_methods_dispatch_and_run():
    for name in all_methods():
        res = _run(name)
        strat = res.extra["strategy"]
        if strat == "sequential":
            # full lower triangular
            for i in range(NUM_TASKS):
                for j in range(NUM_TASKS):
                    assert (res.R[i][j] is not None) == (j <= i)
        elif strat == "joint":
            # only final row filled
            assert all(res.R[NUM_TASKS - 1][j] is not None for j in range(NUM_TASKS))
            assert res.R[0][0] is None
        elif strat == "per_task_oracle":
            # only diagonal filled
            for i in range(NUM_TASKS):
                for j in range(NUM_TASKS):
                    assert (res.R[i][j] is not None) == (i == j)
        else:
            raise AssertionError(f"unknown strategy {strat}")


def test_strategy_tags():
    assert _run("sequential_lora").extra["strategy"] == "sequential"
    assert _run("lora_er").extra["strategy"] == "sequential"
    assert _run("joint").extra["strategy"] == "joint"
    assert _run("per_task_lora_oracle").extra["strategy"] == "per_task_oracle"


if __name__ == "__main__":
    test_all_methods_dispatch_and_run()
    test_strategy_tags()
    print("All runner/method-dispatch tests passed.")
