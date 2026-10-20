"""Phase-2 DoD harness — single-task LoRA training.

Trains LoRA + cosine head on ONE Banking77 task (task 0 by default) for a few
epochs and reports:
  * test accuracy on that task   (DoD target: > 90%)
  * trainable vs total params    (DoD: only LoRA + head trainable)
  * peak VRAM during training     (DoD: < 4 GB)

This is a throwaway single-task check, not the continual loop. Because the
cosine head is indexed 0..K-1 while Banking77 labels are global ids, we map
this task's global class ids to local head rows here. The Phase-3
ContinualTrainer generalizes that mapping across the whole stream.

Run from the repo root:
    python -m scripts.train_single_task
    python -m scripts.train_single_task --task 2 --epochs 5 --lora-r 8
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import yaml

from src.data.streams import build_task_stream
from src.models.backbone import trainable_summary
from src.models.build import build_model
from src.utils.device import get_device, peak_vram_mb, reset_peak_vram
from src.utils.seed import set_seed


def _remap(labels: torch.Tensor, g2l: dict[int, int]) -> torch.Tensor:
    return torch.tensor([g2l[int(x)] for x in labels], device=labels.device)


@torch.no_grad()
def evaluate(model, loader, g2l, device, autocast_dtype) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", batch.get("label")).to(device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits = model(ids, mask)
        pred = logits.argmax(dim=-1)
        correct += (pred == _remap(labels, g2l)).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 single-task LoRA DoD")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"Device: {device}")

    # Mixed precision: bf16 if the GPU supports it (Ampere+), else fp16.
    # GTX 1650 Ti is Turing -> fp16.
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        use_bf16 = torch.cuda.is_bf16_supported() and major >= 8   # bf16 od Ampere (8.0)
        autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        autocast_dtype = None

    stream = build_task_stream(cfg, sanity=False)
    t = args.task
    classes = sorted(stream.new_classes(t))
    g2l = {g: i for i, g in enumerate(classes)}
    print(f"Task {t}: {len(classes)} classes -> head rows 0..{len(classes)-1}")

    method_cfg = {"peft": "lora", "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}
    model = build_model(
        method_cfg, cfg["backbone"]["name"], num_initial_classes=0, pooling="mean"
    )
    # Grow the head to this task's classes, THEN build the optimizer
    # (optimizer must see the expanded Parameter — see CosineHead docstring).
    model.expand_head(len(classes))
    model.to(device)
    print(trainable_summary(model))

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=(autocast_dtype == torch.float16))
    criterion = nn.CrossEntropyLoss()

    train_loader = stream.train_loader(t)
    test_loader = stream.test_loader(t)

    reset_peak_vram()
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = _remap(batch.get("labels", batch.get("label")).to(device), g2l)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model(ids, mask)
                loss = criterion(logits, labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running += loss.item()
        acc = evaluate(model, test_loader, g2l, device, autocast_dtype)
        print(f"  epoch {epoch+1}/{args.epochs}  loss={running/len(train_loader):.4f}  test_acc={acc:.4f}")

    elapsed = time.time() - start
    final_acc = evaluate(model, test_loader, g2l, device, autocast_dtype)
    peak = peak_vram_mb()

    print("\n--- Phase-2 DoD ---")
    print(trainable_summary(model))
    print(f"Final task-{t} test accuracy: {final_acc:.4f}  (target > 0.90)")
    print(f"Training time: {elapsed:.1f}s")
    if device.type == "cuda":
        print(f"Peak VRAM: {peak:.1f} MB  (budget 4096 MB)")
    checks = [final_acc > 0.90]
    if device.type == "cuda":
        checks.append(peak < 4096)
    print("DoD:", "ALL GREEN" if all(checks) else "see numbers above")


if __name__ == "__main__":
    main()
