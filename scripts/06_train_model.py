"""Phase 6 — QLoRA fine-tuning on rented GPU.

Reads:  data/processed/{train,val}.jsonl, models/tokenizer/ (optional)
Writes: models/lora_adapter/, W&B logs

Run on the GPU box, not local Windows:
    python scripts/06_train_model.py --base-model Qwen/Qwen2.5-1.5B
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/lora_adapter"))
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="ryGPT")
    args = parser.parse_args()

    raise NotImplementedError("Phase 6 implementation pending.")


if __name__ == "__main__":
    main()
