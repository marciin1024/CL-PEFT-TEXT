"""Device selection and peak-VRAM tracking helpers.

The project's hardware budget is 4 GB VRAM; every training/eval run should
report peak allocated memory so regressions are caught early.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return cuda if available (and preferred), else cpu."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reset_peak_vram(device: torch.device | None = None) -> None:
    """Reset CUDA peak-memory statistics (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_mb(device: torch.device | None = None) -> float:
    """Peak allocated VRAM in MB since the last reset (0.0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


@contextmanager
def track_peak_vram(label: str = "") -> Iterator[None]:
    """Context manager: resets stats on entry, prints peak on exit.

    Usage:
        with track_peak_vram("task 3 training"):
            train_one_task(...)
    """
    reset_peak_vram()
    try:
        yield
    finally:
        mb = peak_vram_mb()
        tag = f" [{label}]" if label else ""
        print(f"Peak VRAM{tag}: {mb:.1f} MB")
