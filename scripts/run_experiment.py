"""Phase-5 experiment runner.

Runs one or more methods on Banking77 (7x11 CIL), computes metrics, and writes:
  * results/runs/<method>_<dataset>_seed<seed>.json   (full record per method)
  * results/main_table.csv                            (one row per method)

Each JSON records the R matrix, metrics, per-task time/VRAM, the resolved
config, the git commit, and a timestamp — enough to reproduce and to drop
straight into the report.

    python -m scripts.run_experiment --methods lora_er
    python -m scripts.run_experiment --methods sequential_lora,lora_er,joint
    python -m scripts.run_experiment --all
    python -m scripts.run_experiment --all --quick     # tiny Banking77, ~minutes

`--quick` caps Banking77 to a few examples/class so the whole 7-task pipeline
runs in minutes for debugging; drop it for the real (weekend) run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import yaml

from src.cl.methods import all_methods, get_method
from src.cl.runner import pick_autocast, run_method
from src.data.streams import build_task_stream
from src.eval.metrics import compute_metrics
from src.utils.device import get_device
from src.utils.seed import set_seed

RUNS_DIR = os.path.join("results", "runs")
TABLE_PATH = os.path.join("results", "main_table.csv")
TABLE_COLUMNS = [
    "method", "strategy", "dataset", "seed",
    "ACC", "Forgetting", "BWT", "final_cum_acc",
    "total_time_s", "peak_vram_mb",
]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def apply_quick(cfg: dict) -> None:
    """Shrink Banking77 for fast end-to-end debugging."""
    cfg["data"]["max_train_per_class"] = 30
    cfg["data"]["max_val_per_class"] = 10
    cfg["data"]["max_test_per_class"] = 40


def run_record(method: str, cfg: dict, stream, device, autocast_dtype, dataset: str) -> dict:
    spec = get_method(method)
    epochs = spec.get("epochs", cfg.get("train", {}).get("epochs", 3))
    lr = spec.get("lr", cfg.get("train", {}).get("lr", 1e-3))
    print(f"\n=== {method} (strategy={spec['strategy']}, lr={lr}, epochs={epochs}) ===")

    t0 = time.time()
    res = run_method(method, cfg, stream, device, epochs=epochs, lr=lr, autocast_dtype=autocast_dtype)
    wall = time.time() - t0
    metrics = compute_metrics(res)

    record = {
        "method": method,
        "dataset": dataset,
        "seed": cfg.get("seed", 42),
        "epochs": epochs,
        "lr": lr,
        "git": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(wall, 1),
        "R": res.R,
        "metrics": metrics,
        "time_per_task": [round(x, 1) for x in res.time_per_task],
        "peak_vram_per_task": [round(x, 1) for x in res.peak_vram_per_task],
        "num_classes_per_task": res.num_classes_per_task,
        "spec": spec,
    }
    print(f"  ACC={metrics['ACC']} Forgetting={metrics['Forgetting']} "
          f"BWT={metrics['BWT']} | {metrics['total_time_s']}s | {metrics['peak_vram_mb']}MB")
    return record


def save_record(record: dict) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    fname = f"{record['method']}_{record['dataset']}_seed{record['seed']}.json"
    with open(os.path.join(RUNS_DIR, fname), "w") as f:
        json.dump(record, f, indent=2)


def rebuild_table() -> None:
    """Aggregate every JSON in results/runs into results/main_table.csv."""
    if not os.path.isdir(RUNS_DIR):
        return
    rows = []
    for fn in sorted(os.listdir(RUNS_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(RUNS_DIR, fn)) as f:
            rec = json.load(f)
        m = rec["metrics"]
        rows.append({
            "method": rec["method"], "strategy": m["strategy"],
            "dataset": rec["dataset"], "seed": rec["seed"],
            "ACC": m["ACC"], "Forgetting": m["Forgetting"], "BWT": m["BWT"],
            "final_cum_acc": m["final_cum_acc"],
            "total_time_s": m["total_time_s"], "peak_vram_mb": m["peak_vram_mb"],
        })
    os.makedirs("results", exist_ok=True)
    with open(TABLE_PATH, "w", newline="") as f:
        f.write(",".join(TABLE_COLUMNS) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in TABLE_COLUMNS) + "\n")
    print(f"\nWrote {TABLE_PATH} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-5 experiment runner")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--methods", default="lora_er", help="comma-separated method names")
    parser.add_argument("--all", action="store_true", help="run every registry method")
    parser.add_argument("--quick", action="store_true", help="tiny Banking77 for debugging")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.quick:
        apply_quick(cfg)

    set_seed(cfg.get("seed", 42))
    device = get_device()
    autocast_dtype = pick_autocast(device)
    dataset = cfg["data"]["dataset"] + ("_quick" if args.quick else "")
    print(f"Device: {device} | autocast: {autocast_dtype} | dataset: {dataset} | seed: {cfg.get('seed')}")

    methods = all_methods() if args.all else [m.strip() for m in args.methods.split(",")]
    for m in methods:
        get_method(m)  # validate early

    stream = build_task_stream(cfg, sanity=False)
    print(f"tasks: {stream.num_tasks} | classes: {stream.num_classes}")

    for m in methods:
        record = run_record(m, cfg, stream, device, autocast_dtype, dataset)
        save_record(record)
        rebuild_table()  # rebuild after each method so partial runs accumulate

    print("\nPhase-5 run complete.")


if __name__ == "__main__":
    main()
