"""Unit tests for CL metrics on hand-computed R matrices."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.eval.metrics import (
    average_accuracy,
    average_forgetting,
    backward_transfer,
    compute_metrics,
)


@dataclass
class _Res:
    R: list
    cumulative_acc: list
    time_per_task: list
    peak_vram_per_task: list
    extra: dict = field(default_factory=dict)


# Worked example (3 tasks, sequential):
#   R[0]=[.9,  -,  -]
#   R[1]=[.6, .8,  -]
#   R[2]=[.5, .7, .85]
SEQ = [
    [0.9, None, None],
    [0.6, 0.8, None],
    [0.5, 0.7, 0.85],
]


def test_average_accuracy():
    assert abs(average_accuracy(SEQ, "sequential") - (0.5 + 0.7 + 0.85) / 3) < 1e-9


def test_forgetting():
    # task0: max(.9,.6)-.5 = .4 ; task1: .8-.7 = .1 ; mean = .25
    assert abs(average_forgetting(SEQ) - 0.25) < 1e-9


def test_bwt():
    # task0: .5-.9=-.4 ; task1: .7-.8=-.1 ; mean = -.25
    assert abs(backward_transfer(SEQ) - (-0.25)) < 1e-9


def test_joint_only_uses_final_row():
    R = [[None, None], [0.8, 0.9]]
    assert abs(average_accuracy(R, "joint") - 0.85) < 1e-9


def test_oracle_uses_diagonal():
    R = [[0.92, None], [None, 0.99]]
    assert abs(average_accuracy(R, "per_task_oracle") - 0.955) < 1e-9


def test_compute_metrics_strategy_aware():
    res = _Res(R=SEQ, cumulative_acc=[0.6, 0.55, 0.6833],
               time_per_task=[10, 12, 11], peak_vram_per_task=[1300, 1300, 1350])
    m = compute_metrics(res)
    assert m["strategy"] == "sequential"
    assert m["Forgetting"] is not None and m["BWT"] is not None
    assert m["peak_vram_mb"] == 1350.0

    res.extra = {"strategy": "joint"}
    res.R = [[None, None], [0.8, 0.9]]
    res.cumulative_acc = [0.85]
    m2 = compute_metrics(res)
    assert m2["Forgetting"] is None and m2["BWT"] is None
    assert abs(m2["ACC"] - 0.85) < 1e-9


if __name__ == "__main__":
    test_average_accuracy()
    test_forgetting()
    test_bwt()
    test_joint_only_uses_final_row()
    test_oracle_uses_diagonal()
    test_compute_metrics_strategy_aware()
    print("All metrics tests passed.")
