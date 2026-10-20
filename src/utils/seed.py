"""Global reproducibility utilities.

One call to `set_seed(seed)` at the top of every run covers Python's
`random`, NumPy, PyTorch (CPU + CUDA) and sets the cuDNN deterministic
flags. Also provides `seed_worker` for DataLoader workers so that
shuffling is reproducible across runs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Seed every RNG the project touches.

    Args:
        seed: the global seed (logged with every experiment).
        deterministic_cudnn: if True, force cuDNN into deterministic mode.
            Slightly slower, fully reproducible. Disable only for quick
            throwaway runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Hash-based ops (e.g. set/dict iteration in some libs)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for torch DataLoader — derives a per-worker seed
    from torch's initial seed so multi-worker loading stays reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A torch.Generator for DataLoader(shuffle=True, generator=...) so the
    shuffle order is governed by the global seed, not global state."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
