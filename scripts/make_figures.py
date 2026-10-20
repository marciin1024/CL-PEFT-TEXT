"""Phase-6 figures — generated from results/runs/*.json (no training).

Produces, into results/figures/:
  * main_acc_bar.png        — final ACC per method (references styled apart)
  * forgetting_bar.png      — Average Forgetting per sequential method
  * avg_accuracy_curve.png  — avg accuracy over seen tasks vs task index,
                              one line per sequential method, references as
                              horizontal lines
  * heatmap_<method>.png    — the R matrix for each sequential method

The per-task accuracy trajectory is derived from R (mean of the filled part of
row i = average accuracy over tasks seen after task i), so no extra logging is
needed beyond the saved R matrix.

    python -m scripts.make_figures
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

RUNS_DIR = os.path.join("results", "runs")
FIG_DIR = os.path.join("results", "figures")

REFERENCE = {"joint", "per_task_lora_oracle"}
ORDER = [
    "naive_ft", "linear_probe", "sequential_lora", "sequential_adapter",
    "lora_er", "joint", "per_task_lora_oracle",
]
METHOD_COLOR = "#1f77b4"
REF_COLOR = "#9467bd"


def load_records() -> dict:
    recs = {}
    for fn in glob.glob(os.path.join(RUNS_DIR, "*.json")):
        with open(fn) as f:
            r = json.load(f)
        recs[r["method"]] = r
    return recs


def ordered(recs: dict) -> list:
    return [m for m in ORDER if m in recs] + [m for m in recs if m not in ORDER]


def _avg_acc_trajectory(R):
    """Mean of the filled lower-triangular part of each row i =
    average accuracy over the tasks seen after task i."""
    traj = []
    for i in range(len(R)):
        row = [R[i][j] for j in range(i + 1) if R[i][j] is not None]
        traj.append(sum(row) / len(row) if row else None)
    return traj


def fig_main_acc(recs):
    names = ordered(recs)
    accs = [recs[m]["metrics"]["ACC"] for m in names]
    colors = [REF_COLOR if m in REFERENCE else METHOD_COLOR for m in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(names)), accs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Average Accuracy (ACC)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Final accuracy on Banking77 (7x11 class-incremental)")
    for i, a in enumerate(accs):
        ax.text(i, a + 0.012, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(handles=[
        Patch(color=METHOD_COLOR, label="method"),
        Patch(color=REF_COLOR, label="reference (upper bound / oracle)"),
    ], loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "main_acc_bar.png"), dpi=150)
    plt.close(fig)


def fig_forgetting(recs):
    names = [m for m in ordered(recs)
             if recs[m]["metrics"].get("Forgetting") is not None]
    if not names:
        return
    vals = [recs[m]["metrics"]["Forgetting"] for m in names]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(names)), vals, color="#d62728")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Average Forgetting")
    ax.set_ylim(0, 1.05)
    ax.set_title("Average forgetting (lower is better)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "forgetting_bar.png"), dpi=150)
    plt.close(fig)


def fig_trajectory(recs):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in ordered(recs):
        r = recs[m]
        if r["metrics"]["strategy"] != "sequential":
            continue
        traj = _avg_acc_trajectory(r["R"])
        xs = list(range(1, len(traj) + 1))
        ax.plot(xs, traj, marker="o", label=m)
    T = len(next(iter(recs.values()))["R"])
    for m in recs:
        if m in REFERENCE:
            acc = recs[m]["metrics"]["ACC"]
            ax.axhline(acc, ls="--", color=REF_COLOR, lw=1, alpha=0.8)
            ax.text(T, acc, f" {m}", va="center", ha="left", fontsize=8, color=REF_COLOR)
    ax.set_xlabel("after training task #")
    ax.set_ylabel("avg accuracy over seen tasks")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, T + 1))
    ax.set_title("Average accuracy over the task stream")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "avg_accuracy_curve.png"), dpi=150)
    plt.close(fig)


def fig_heatmap(method, record):
    R = record["R"]
    T = len(R)
    M = np.full((T, T), np.nan)
    for i in range(T):
        for j in range(T):
            if R[i][j] is not None:
                M[i][j] = R[i][j]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    for i in range(T):
        for j in range(T):
            if not np.isnan(M[i][j]):
                ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center",
                        color="white" if M[i][j] < 0.6 else "black", fontsize=7)
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xlabel("evaluated on task j")
    ax.set_ylabel("after training task i")
    ax.set_title(f"R matrix — {method}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"heatmap_{method}.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    recs = load_records()
    if not recs:
        print(f"No result JSONs found in {RUNS_DIR}. Run scripts.run_experiment first.")
        return

    fig_main_acc(recs)
    fig_forgetting(recs)
    fig_trajectory(recs)
    for m, r in recs.items():
        if r["metrics"]["strategy"] == "sequential":
            fig_heatmap(m, r)

    written = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FIG_DIR, "*.png")))
    print(f"Wrote {len(written)} figures to {FIG_DIR}:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
