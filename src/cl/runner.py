"""Unified method runner.

`run_method(name, cfg, stream, device, ...)` looks up the method spec and
dispatches to the right strategy, always returning a `CLResults`:

  * sequential      -> ContinualTrainer.run(): full lower-triangular R
  * joint           -> train once on pooled data; fills the LAST row of R with
                       per-task final accuracy (the CIL upper bound)
  * per_task_oracle -> a fresh adapter per task; fills the DIAGONAL of R with
                       each task's own-adapter accuracy (Task-IL reference)

`extra["strategy"]` records which interpretation applies, so Phase-5 metrics
know that Forgetting/BWT are only defined for the sequential strategy.
"""

from __future__ import annotations

import gc
import time
from typing import Callable, Optional

import torch
import torch.nn as nn

from src.cl.buffer import ReservoirBuffer
from src.cl.label_map import LabelMapper
from src.cl.methods import get_method
from src.cl.trainer import CLResults, ContinualTrainer, _get_labels
from src.eval.cl_eval import evaluate_cil
from src.models.build import build_model
from src.utils.device import peak_vram_mb, reset_peak_vram


def pick_autocast(device: torch.device) -> Optional[torch.dtype]:
    """bf16 only on Ampere+ (sm 8.0); fp16 on older CUDA GPUs; None on CPU.
    GTX 1650 Ti is Turing (7.5) -> fp16."""
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability()
    use_bf16 = torch.cuda.is_bf16_supported() and major >= 8
    return torch.bfloat16 if use_bf16 else torch.float16


def _free_cuda() -> None:
    """Release a finished run's model so VRAM does not accumulate across
    successive runs (PEFT/adapter hooks create reference cycles that the
    refcounting GC alone will not collect promptly)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _backbone(cfg) -> str:
    return cfg["backbone"]["name"]


def _pooling(cfg) -> str:
    return cfg.get("train", {}).get("pooling", "mean")


def _scale(cfg) -> float:
    return cfg.get("train", {}).get("cosine_scale", 16.0)


def _train_one(model, loader, mapper, device, epochs, lr, autocast_dtype, log, tag):
    """Train a model on a single loader (no replay). Returns (seconds, peak_MB)."""
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=(autocast_dtype == torch.float16))
    criterion = nn.CrossEntropyLoss()
    reset_peak_vram()
    start = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = _get_labels(batch).to(device)
            idx = mapper.to_idx(labels)
            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                loss = criterion(model(ids, mask), idx)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running += loss.item()
        log(f"  {tag} epoch {epoch+1}/{epochs} loss={running/max(len(loader),1):.4f}")
    return time.time() - start, peak_vram_mb()


# ---- strategies --------------------------------------------------------

def _run_sequential(spec, cfg, stream, device, epochs, lr, autocast_dtype, log) -> CLResults:
    model = build_model(spec, _backbone(cfg), num_initial_classes=0, pooling=_pooling(cfg), scale=_scale(cfg))
    model.to(device)
    buffer = None
    if spec.get("replay"):
        pad_id = getattr(stream.tokenizer, "pad_token_id", 1) or 1
        buffer = ReservoirBuffer(
            capacity=spec.get("buffer_capacity", 500), pad_token_id=pad_id, seed=cfg.get("seed", 42)
        )
    trainer = ContinualTrainer(
        model=model, stream=stream, device=device, epochs=epochs, lr=lr,
        autocast_dtype=autocast_dtype, buffer=buffer,
        replay_ratio=spec.get("replay_ratio", 1.0), log=log,
    )
    res = trainer.run()
    res.extra["strategy"] = "sequential"
    return res


def _run_joint(spec, cfg, stream, device, epochs, lr, autocast_dtype, log) -> CLResults:
    T = stream.num_tasks
    mapper = LabelMapper()
    for t in range(T):                       # register ALL classes up front
        mapper.register(sorted(stream.new_classes(t)))

    model = build_model(spec, _backbone(cfg), num_initial_classes=len(mapper),
                        pooling=_pooling(cfg), scale=_scale(cfg))
    model.to(device)

    log("joint: training once on pooled data")
    elapsed, peak = _train_one(
        model, stream.joint_train_loader(), mapper, device, epochs, lr, autocast_dtype, log, "joint"
    )

    R = [[None] * T for _ in range(T)]       # fill only the final row
    for j in range(T):
        R[T - 1][j] = evaluate_cil(model, stream.test_loader(j), mapper, device, autocast_dtype)
    cum = evaluate_cil(model, stream.cumulative_test_loader(T - 1), mapper, device, autocast_dtype)
    log(f"joint done: cum_acc={cum:.4f} | per-task={[round(R[T-1][j],3) for j in range(T)]}")

    return CLResults(
        R=R, cumulative_acc=[cum], time_per_task=[elapsed], peak_vram_per_task=[peak],
        num_classes_per_task=[len(mapper)], extra={"strategy": "joint"},
    )


def _run_per_task_oracle(spec, cfg, stream, device, epochs, lr, autocast_dtype, log) -> CLResults:
    T = stream.num_tasks
    R = [[None] * T for _ in range(T)]       # fill only the diagonal
    times, peaks = [], []
    for t in range(T):
        local = LabelMapper()                # fresh head per task: only its classes
        local.register(sorted(stream.new_classes(t)))
        model = build_model(spec, _backbone(cfg), num_initial_classes=len(local),
                            pooling=_pooling(cfg), scale=_scale(cfg))
        model.to(device)
        elapsed, peak = _train_one(
            model, stream.train_loader(t), local, device, epochs, lr, autocast_dtype, log, f"oracle t{t}"
        )
        # eval restricted to this task's own head (Task-IL, oracle task-id)
        R[t][t] = evaluate_cil(model, stream.test_loader(t), local, device, autocast_dtype)
        times.append(elapsed)
        peaks.append(peak)
        log(f"oracle task {t}: acc={R[t][t]:.4f} | {elapsed:.1f}s | {peak:.0f}MB")
        del model
        _free_cuda()

    diag = [R[t][t] for t in range(T)]
    return CLResults(
        R=R, cumulative_acc=[sum(diag) / T], time_per_task=times, peak_vram_per_task=peaks,
        num_classes_per_task=[len(stream.new_classes(t)) for t in range(T)],
        extra={"strategy": "per_task_oracle"},
    )


_STRATEGIES: dict[str, Callable] = {
    "sequential": _run_sequential,
    "joint": _run_joint,
    "per_task_oracle": _run_per_task_oracle,
}


def run_method(
    name: str,
    cfg: dict,
    stream,
    device: torch.device,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    autocast_dtype: Optional[torch.dtype] = None,
    overrides: Optional[dict] = None,
    log: Callable[[str], None] = print,
) -> CLResults:
    spec = get_method(name)
    if overrides:
        spec.update(overrides)  # ablation knobs: lora_r, lora_alpha, buffer_capacity, ...
    strategy = spec.get("strategy", "sequential")
    fn = _STRATEGIES[strategy]
    res = fn(spec, cfg, stream, device, epochs, lr, autocast_dtype, log)
    _free_cuda()
    return res
