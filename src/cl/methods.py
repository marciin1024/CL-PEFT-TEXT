"""Method registry — config name -> how the model is built and trained.

Each entry is a flat spec consumed by `build_model` (the `peft` key + PEFT
hyperparameters) and by the runner (`strategy`, replay, buffer capacity). The
`strategy` field is the dispatch switch:

  "sequential"      -> task-by-task via ContinualTrainer (full R matrix)
  "joint"           -> pool all tasks, train once (CIL upper bound)
  "per_task_oracle" -> a SEPARATE adapter per task, task-id known at eval
                       (Task-IL reference ONLY — not a method we compete with)

Per-method `lr`/`epochs` matter: full fine-tuning needs a far smaller lr than
PEFT (updating 125M weights at 1e-3 diverges), so naive_ft uses 2e-5 and more
epochs while PEFT/linear use 1e-3. The runner reads these (CLI flags override).
"""

from __future__ import annotations

from typing import Any, Dict, List

METHODS: Dict[str, Dict[str, Any]] = {
    # --- single-model CIL methods (sequential loop) --------------------
    "naive_ft": {            # lower bound: unfreeze whole backbone, no replay
        "strategy": "sequential",
        "peft": "full",
        "replay": False,
        "lr": 2e-5,          # full FT needs a small lr
        "epochs": 5,
    },
    "linear_probe": {        # stability reference: frozen base, head only
        "strategy": "sequential",
        "peft": "linear",
        "replay": False,
        "lr": 1e-3,
        "epochs": 3,
    },
    "sequential_lora": {     # one shared LoRA adapter, updated task-by-task
        "strategy": "sequential",
        "peft": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "replay": False,
        "lr": 1e-3,
        "epochs": 3,
    },
    "sequential_adapter": {  # bottleneck adapters (custom, Option B)
        "strategy": "sequential",
        "peft": "adapter",
        "reduction_factor": 16,
        "replay": False,
        "lr": 1e-3,
        "epochs": 3,
    },
    "lora_er": {             # expected best single-model method
        "strategy": "sequential",
        "peft": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "replay": True,
        "buffer_capacity": 500,
        "replay_ratio": 1.0,
        "lr": 1e-3,
        "epochs": 3,
    },
    # --- references ----------------------------------------------------
    "joint": {               # upper bound: pool all tasks, train once
        "strategy": "joint",
        "peft": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "replay": False,
        "lr": 1e-3,
        "epochs": 3,
    },
    "per_task_lora_oracle": {  # Task-IL reference ONLY (label honestly)
        "strategy": "per_task_oracle",
        "peft": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "replay": False,
        "lr": 1e-3,
        "epochs": 3,
    },
}


def get_method(name: str) -> Dict[str, Any]:
    if name not in METHODS:
        raise KeyError(f"Unknown method '{name}'. Known: {sorted(METHODS)}")
    return dict(METHODS[name])


def all_methods() -> List[str]:
    return list(METHODS.keys())
