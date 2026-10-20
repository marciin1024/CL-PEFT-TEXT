"""Ablation figures — generated from results/ablations/*_table.csv (no training).

Produces, into results/figures/:
  * ablation_rank.png   — ACC vs LoRA rank r
  * ablation_buffer.png — ACC vs replay buffer capacity
  * ablation_order.png  — ACC per task-order seed, with mean +/- std

    python -m scripts.make_ablation_figures
"""

from __future__ import annotations

import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABL_DIR = os.path.join("results", "ablations")
FIG_DIR = os.path.join("results", "figures")
ACCENT = "#1f77b4"


def _read(name):
    path = os.path.join(ABL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_rank():
    rows = _read("rank_table.csv")
    if not rows:
        return False
    rs = [int(r["r"]) for r in rows]
    acc = [float(r["ACC"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(rs, acc, marker="o", color=ACCENT, lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(rs)
    ax.set_xticklabels([str(r) for r in rs])
    ax.set_xlabel("LoRA rank r  (log2 scale; alpha = 2r)")
    ax.set_ylabel("Average Accuracy (ACC)")
    ax.set_title("LoRA rank ablation — lora_er, Banking77 (7x11)")
    for x, y in zip(rs, acc):
        ax.text(x, y + 0.004, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(min(acc) - 0.03, max(acc) + 0.03)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ablation_rank.png"), dpi=150)
    plt.close(fig)
    return True


def fig_buffer():
    rows = _read("buffer_table.csv")
    if not rows:
        return False
    caps = [int(r["capacity"]) for r in rows]
    acc = [float(r["ACC"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(caps, acc, marker="o", color=ACCENT, lw=2)
    ax.set_xscale("log")
    ax.set_xticks(caps)
    ax.set_xticklabels([str(c) for c in caps])
    ax.set_xlabel("replay buffer capacity  (log scale)")
    ax.set_ylabel("Average Accuracy (ACC)")
    ax.set_title("Replay buffer-size ablation — lora_er, Banking77 (7x11)")
    for x, y in zip(caps, acc):
        ax.text(x, y + 0.008, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(min(acc) - 0.05, max(acc) + 0.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ablation_buffer.png"), dpi=150)
    plt.close(fig)
    return True


def fig_order():
    rows = _read("order_table.csv")
    if not rows:
        return False
    seeds = [int(r["class_order_seed"]) for r in rows]
    acc = [float(r["ACC"]) for r in rows]
    mean = statistics.mean(acc)
    std = statistics.pstdev(acc) if len(acc) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.bar(range(len(seeds)), acc, color=ACCENT, width=0.55)
    ax.axhline(mean, ls="--", color="#d62728", lw=1.5,
               label=f"mean = {mean:.3f} +/- {std:.3f}")
    ax.axhspan(mean - std, mean + std, color="#d62728", alpha=0.12)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Average Accuracy (ACC)")
    ax.set_title("Task-order sensitivity — lora_er, Banking77 (7x11)")
    for i, y in enumerate(acc):
        ax.text(i, y + 0.004, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(min(acc) - 0.05, max(acc) + 0.04)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ablation_order.png"), dpi=150)
    plt.close(fig)
    return True


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    done = []
    if fig_rank():
        done.append("ablation_rank.png")
    if fig_buffer():
        done.append("ablation_buffer.png")
    if fig_order():
        done.append("ablation_order.png")
    if not done:
        print(f"No ablation CSVs found in {ABL_DIR}. Run scripts.run_ablation first.")
        return
    print(f"Wrote {len(done)} ablation figures to {FIG_DIR}:")
    for d in done:
        print(f"  {d}")


if __name__ == "__main__":
    main()
