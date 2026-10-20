"""Phase-0 smoke test.

Loads the frozen backbone, tokenizes one batch, runs a forward pass on the
selected device and prints peak VRAM. DoD: peak VRAM < 4 GB.

Run from the repo root:
    python -m scripts.smoke
    python -m scripts.smoke --model microsoft/deberta-v3-base --fp16
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModel, AutoTokenizer

from src.utils.device import get_device, peak_vram_mb, reset_peak_vram
from src.utils.seed import set_seed

SAMPLE_UTTERANCES = [
    "I would like to know why my card payment was declined",
    "How do I top up my account using a bank transfer?",
    "My card has not arrived yet, what should I do?",
    "Can you tell me the exchange rate you apply for euros?",
    "I think someone made a payment with my card without my consent",
    "How long does a refund usually take to appear?",
    "I want to close my account permanently",
    "Why was I charged a fee on my last withdrawal?",
] * 4  # batch of 32


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-0 smoke test")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--fp16", action="store_true", help="run the forward pass in autocast")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"Total VRAM: {total_mb:.0f} MB")

    print(f"Loading tokenizer + backbone: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    # Freeze everything — this project never trains the base weights directly.
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone params: {n_params/1e6:.1f}M | trainable: {n_trainable}")

    batch = tokenizer(
        SAMPLE_UTTERANCES,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    ).to(device)

    reset_peak_vram()
    with torch.no_grad():
        if args.fp16 and device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(**batch)
        else:
            out = model(**batch)

    hidden = out.last_hidden_state
    print(f"Forward OK — last_hidden_state: {tuple(hidden.shape)}")

    peak = peak_vram_mb()
    print(f"Peak VRAM: {peak:.1f} MB")
    if device.type == "cuda":
        budget_mb = 4 * 1024
        status = "OK (< 4 GB)" if peak < budget_mb else "EXCEEDED 4 GB BUDGET"
        print(f"Budget check: {status}")
    else:
        print("Budget check: skipped (CPU run)")


if __name__ == "__main__":
    main()
