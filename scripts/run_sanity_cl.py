"""Phase-3/4 DoD harness — run any registry method end-to-end and print R.

Dispatches through the unified runner, so all 7 methods work from one name.
Default: sequential_lora on the AG News sanity stream (2 tasks x 2 classes),
subsampled per class so it finishes in ~1 minute.

    python -m scripts.run_sanity_cl
    python -m scripts.run_sanity_cl --method lora_er
    python -m scripts.run_sanity_cl --method joint
    python -m scripts.run_sanity_cl --all          # run every method
    python -m scripts.run_sanity_cl --method sequential_lora --full   # Banking77
"""

from __future__ import annotations

import argparse

import yaml

from src.cl.methods import all_methods, get_method
from src.cl.runner import pick_autocast, run_method
from src.data.streams import build_task_stream
from src.utils.device import get_device
from src.utils.seed import set_seed


def print_matrix(res, log=print):
    T = len(res.R)
    strat = res.extra.get("strategy", "sequential")
    log(f"\nAccuracy matrix R[i][j]  (strategy: {strat}):")
    log("        " + " ".join(f"  t{j}  " for j in range(T)))
    for i in range(T):
        cells = [f"{res.R[i][j]:.3f}" if res.R[i][j] is not None else "  -  " for j in range(T)]
        log(f"  t{i}  | " + " ".join(cells))
    log("Cumulative (CIL) acc: " + ", ".join(f"{a:.3f}" for a in res.cumulative_acc))


def run_one(method, cfg, stream, device, autocast_dtype, epochs, lr):
    print(f"\n=== {method} ===")
    res = run_method(method, cfg, stream, device,
                     epochs=epochs, lr=lr, autocast_dtype=autocast_dtype)
    print_matrix(res)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-3/4 CL harness")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--method", default="sequential_lora")
    parser.add_argument("--all", action="store_true", help="run every registry method")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--full", action="store_true", help="use Banking77 instead of sanity")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    autocast_dtype = pick_autocast(device)
    print(f"Device: {device} | autocast: {autocast_dtype}")

    stream = build_task_stream(cfg, sanity=not args.full)
    print(f"tasks: {stream.num_tasks} | classes: {stream.num_classes}")

    methods = all_methods() if args.all else [args.method]
    for m in methods:
        get_method(m)  # validate name early
    for m in methods:
        run_one(m, cfg, stream, device, autocast_dtype, args.epochs, args.lr)

    print("\nPhase-4 DoD:", "ALL GREEN (methods dispatched & ran)")


if __name__ == "__main__":
    main()
