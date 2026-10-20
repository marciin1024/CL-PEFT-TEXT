"""Continual-learning metrics computed from the accuracy matrix R.

R[i][j] = accuracy on task j's test set after finishing task i (CIL eval).
Conventions follow Lopez-Paz & Ranzato (GEM) / Chaudhry et al.

Metrics (sequential strategy):
  * ACC (Average Accuracy):
        ACC = mean_j R[T-1][j]
    accuracy on every task after the final task.
  * Average Forgetting:
        F = mean_{j<T-1} ( max_{i in [j, T-2]} R[i][j] - R[T-1][j] )
    per old task, its best-ever accuracy minus its final accuracy (>=0; higher
    is worse).
  * BWT (Backward Transfer):
        BWT = mean_{j<T-1} ( R[T-1][j] - R[j][j] )
    how learning later tasks changed old-task accuracy (negative = forgetting).

Strategy-specific reductions:
  * joint  -> only the final row is filled; ACC = mean of it (the CIL upper
    bound). Forgetting/BWT are undefined (no per-task-over-time trace).
  * per_task_oracle -> only the diagonal is filled; ACC = mean of it (the
    Task-IL reference). Forgetting/BWT undefined.

FWT (Forward Transfer) is intentionally omitted: it requires evaluating a
future task *before* training it, but under an expanding cosine head the
future task's classes are not yet in the head, so the quantity is ill-defined
in this expanding-head CIL setup. We document this rather than report a
misleading number.
"""

from __future__ import annotations

from typing import Dict, List, Optional

Matrix = List[List[Optional[float]]]


def _final_row_mean(R: Matrix) -> float:
    T = len(R)
    vals = [R[T - 1][j] for j in range(T) if R[T - 1][j] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _diagonal_mean(R: Matrix) -> float:
    T = len(R)
    vals = [R[t][t] for t in range(T) if R[t][t] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def average_accuracy(R: Matrix, strategy: str = "sequential") -> float:
    if strategy == "per_task_oracle":
        return _diagonal_mean(R)
    return _final_row_mean(R)  # sequential and joint both use the final row


def average_forgetting(R: Matrix) -> float:
    """Sequential only. Returns 0.0 for T<2."""
    T = len(R)
    if T < 2:
        return 0.0
    per_task = []
    for j in range(T - 1):
        history = [R[i][j] for i in range(j, T - 1) if R[i][j] is not None]
        final = R[T - 1][j]
        if history and final is not None:
            per_task.append(max(history) - final)
    return sum(per_task) / len(per_task) if per_task else 0.0


def backward_transfer(R: Matrix) -> float:
    """Sequential only. Returns 0.0 for T<2."""
    T = len(R)
    if T < 2:
        return 0.0
    per_task = [
        R[T - 1][j] - R[j][j]
        for j in range(T - 1)
        if R[T - 1][j] is not None and R[j][j] is not None
    ]
    return sum(per_task) / len(per_task) if per_task else 0.0


def compute_metrics(results) -> Dict[str, Optional[float]]:
    """Build the metric dict for a CLResults object, strategy-aware."""
    R = results.R
    strategy = results.extra.get("strategy", "sequential")
    out: Dict[str, Optional[float]] = {
        "strategy": strategy,
        "ACC": round(average_accuracy(R, strategy), 4),
    }
    if strategy == "sequential":
        out["Forgetting"] = round(average_forgetting(R), 4)
        out["BWT"] = round(backward_transfer(R), 4)
    else:
        out["Forgetting"] = None
        out["BWT"] = None
    out["final_cum_acc"] = round(results.cumulative_acc[-1], 4) if results.cumulative_acc else None
    out["total_time_s"] = round(sum(results.time_per_task), 1)
    out["peak_vram_mb"] = round(max(results.peak_vram_per_task), 1) if results.peak_vram_per_task else 0.0
    return out
