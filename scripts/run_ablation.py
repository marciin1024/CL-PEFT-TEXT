"""Phase-6 ablations — all on the headline method `lora_er`.

Three ablations from the proposal:
  * rank   — LoRA rank r in {2,4,8,16,32}, with alpha = 2*r so the LoRA scaling
             alpha/r stays constant (r=8 reproduces the main-table point).
  * buffer — replay buffer capacity in {100,200,500,1000} (500 = main table).
  * order  — task-order sensitivity: rerun with several class_order_seed values
             (different class partitions, same training seed) -> mean +/- std,
             i.e. error bars for the headline method.

Writes per-point JSON to results/ablations/runs/ and a CSV per ablation to
results/ablations/<ablation>_table.csv. Incremental, so a partial run can be
resumed; just re-run with the remaining points.

    python -m scripts.run_ablation --ablation rank
    python -m scripts.run_ablation --ablation buffer
    python -m scripts.run_ablation --ablation order
    python -m scripts.run_ablation --ablation all --quick   # fast smoke test

Each point is a full lora_er run (~40 min on a GTX 1650 Ti at full data), so a
single ablation is a few hours. Use --quick (tiny Banking77 + 2 points each) to
verify the script first; run the real ablations on separate nights.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import yaml

from src.cl.runner import pick_autocast, run_method
from src.data.streams import build_task_stream
from src.eval.metrics import compute_metrics
from src.utils.device import get_device
from src.utils.seed import set_seed
from scripts.run_experiment import apply_quick, git_hash

ABL_DIR = os.path.join("results", "ablations")
RUNS_DIR = os.path.join(ABL_DIR, "runs")
METHOD = "lora_er"

DEFAULTS = {
    "rank": [2, 4, 8, 16, 32],
    "buffer": [100, 200, 500, 1000],
    "order": [0, 1, 2],
}
QUICK = {"rank": [2, 8], "buffer": [100, 500], "order": [0, 1]}

METRIC_KEYS = ["ACC", "Forgetting", "BWT", "final_cum_acc", "total_time_s", "peak_vram_mb"]


def _save_record(ablation, tag, cfg, overrides, res, m):
    os.makedirs(RUNS_DIR, exist_ok=True)
    rec = {
        "ablation": ablation, "tag": tag, "method": METHOD,
        "overrides": overrides, "seed": cfg.get("seed"),
        "class_order_seed": cfg["data"].get("class_order_seed"),
        "git": git_hash(), "R": res.R, "metrics": m,
    }
    with open(os.path.join(RUNS_DIR, f"{ablation}_{tag}.json"), "w") as f:
        json.dump(rec, f, indent=2)


def _save_table(ablation, rows, varied_col):
    os.makedirs(ABL_DIR, exist_ok=True)
    cols = [varied_col] + METRIC_KEYS
    path = os.path.join(ABL_DIR, f"{ablation}_table.csv")
    with open(path, "w", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    print(f"Wrote {path} ({len(rows)} rows)")


def _point(cfg, stream, device, ac, overrides, epochs, lr, label):
    print(f"\n--- {label} ---")
    res = run_method(METHOD, cfg, stream, device, epochs=epochs, lr=lr,
                     autocast_dtype=ac, overrides=overrides)
    m = compute_metrics(res)
    print(f"  {label}: ACC={m['ACC']} F={m['Forgetting']} BWT={m['BWT']} "
          f"| {m['total_time_s']}s | {m['peak_vram_mb']}MB")
    return res, m


def ablate_rank(cfg, stream, device, ac, epochs, lr, ranks):
    rows = []
    for r in ranks:
        ov = {"lora_r": r, "lora_alpha": 2 * r}
        res, m = _point(cfg, stream, device, ac, ov, epochs, lr, f"rank r={r} (alpha={2*r})")
        rows.append({"r": r, **{k: m[k] for k in METRIC_KEYS}})
        _save_record("rank", f"r{r}", cfg, ov, res, m)
        _save_table("rank", rows, "r")


def ablate_buffer(cfg, stream, device, ac, epochs, lr, caps):
    rows = []
    for c in caps:
        ov = {"buffer_capacity": c}
        res, m = _point(cfg, stream, device, ac, ov, epochs, lr, f"buffer capacity={c}")
        rows.append({"capacity": c, **{k: m[k] for k in METRIC_KEYS}})
        _save_record("buffer", f"cap{c}", cfg, ov, res, m)
        _save_table("buffer", rows, "capacity")


def ablate_order(base_cfg, device, ac, epochs, lr, seeds, quick):
    rows = []
    accs = []
    for s in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["class_order_seed"] = s
        stream = build_task_stream(cfg, sanity=False)
        res, m = _point(cfg, stream, device, ac, None, epochs, lr, f"class_order_seed={s}")
        rows.append({"class_order_seed": s, **{k: m[k] for k in METRIC_KEYS}})
        accs.append(m["ACC"])
        _save_record("order", f"seed{s}", cfg, {}, res, m)
        _save_table("order", rows, "class_order_seed")
    if len(accs) >= 2:
        mean = statistics.mean(accs)
        std = statistics.pstdev(accs)
        print(f"\norder ablation: ACC = {mean:.4f} +/- {std:.4f} over {len(accs)} seeds")


def main():
    parser = argparse.ArgumentParser(description="Phase-6 ablations on lora_er")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--ablation", choices=["rank", "buffer", "order", "all"], default="rank")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.quick:
        apply_quick(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    ac = pick_autocast(device)
    grid = QUICK if args.quick else DEFAULTS
    print(f"Device: {device} | autocast: {ac} | quick={args.quick} | ablation={args.ablation}")

    todo = ["rank", "buffer", "order"] if args.ablation == "all" else [args.ablation]

    # rank/buffer reuse one stream (fixed class order); order rebuilds per seed.
    base_stream = None
    if "rank" in todo or "buffer" in todo:
        base_stream = build_task_stream(cfg, sanity=False)
        print(f"tasks: {base_stream.num_tasks} | classes: {base_stream.num_classes}")

    if "rank" in todo:
        ablate_rank(cfg, base_stream, device, ac, args.epochs, args.lr, grid["rank"])
    if "buffer" in todo:
        ablate_buffer(cfg, base_stream, device, ac, args.epochs, args.lr, grid["buffer"])
    if "order" in todo:
        ablate_order(cfg, device, ac, args.epochs, args.lr, grid["order"], args.quick)

    print("\nAblation run complete.")


if __name__ == "__main__":
    main()
