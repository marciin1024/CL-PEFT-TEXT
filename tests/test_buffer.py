"""Unit tests for the reservoir replay buffer (offline, CPU)."""

from __future__ import annotations

import torch

from src.cl.buffer import ReservoirBuffer


def _batch(bsz, seqlen, label_start, n_labels):
    input_ids = torch.randint(5, 100, (bsz, seqlen))
    attention_mask = torch.ones(bsz, seqlen, dtype=torch.long)
    labels = torch.randint(label_start, label_start + n_labels, (bsz,))
    return input_ids, attention_mask, labels


def test_capacity_is_respected():
    buf = ReservoirBuffer(capacity=100, seed=0)
    for _ in range(20):
        buf.add_batch(*_batch(32, 10, 0, 10))  # 640 samples >> capacity
    assert len(buf) == 100
    assert buf.n_seen == 640


def test_class_distribution_is_sane():
    buf = ReservoirBuffer(capacity=200, seed=0)
    # feed 4 classes uniformly; reservoir should keep all 4 represented
    for _ in range(40):
        buf.add_batch(*_batch(32, 8, 0, 4))
    dist = buf.class_distribution()
    assert set(dist.keys()) == {0, 1, 2, 3}
    counts = list(dist.values())
    # no class should be wildly over/under-represented (loose bound)
    assert min(counts) > 0 and max(counts) < 0.7 * len(buf)


def test_sample_shapes_and_padding():
    buf = ReservoirBuffer(capacity=50, pad_token_id=1, seed=0)
    # variable-length sequences (trimmed by attention mask)
    ids = torch.randint(5, 100, (4, 12))
    mask = torch.zeros(4, 12, dtype=torch.long)
    for i, length in enumerate([3, 7, 12, 5]):
        mask[i, :length] = 1
    buf.add_batch(ids, mask, torch.tensor([0, 1, 2, 3]))

    out = buf.sample(4)
    assert out["input_ids"].shape[0] == 4
    assert out["attention_mask"].shape == out["input_ids"].shape
    # rows where mask==0 must hold the pad id
    assert torch.all(out["input_ids"][out["attention_mask"] == 0] == 1)


def test_sample_caps_at_size():
    buf = ReservoirBuffer(capacity=50, seed=0)
    buf.add_batch(*_batch(3, 6, 0, 3))
    out = buf.sample(10)  # ask more than stored
    assert out["labels"].numel() == 3


def test_empty_buffer():
    buf = ReservoirBuffer(capacity=10)
    assert buf.is_empty() and len(buf) == 0


if __name__ == "__main__":
    test_capacity_is_respected()
    test_class_distribution_is_sane()
    test_sample_shapes_and_padding()
    test_sample_caps_at_size()
    test_empty_buffer()
    print("All buffer tests passed.")
