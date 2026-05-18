"""Phase 5 — train a custom BPE tokenizer, compare compression against base.

STOP point. Review eval/tokenizer_compression.md; decide whether to extend
base model vocab (per ADR-005: ≥ 1.5x ratio threshold).

Reads:  data/processed/train.jsonl
Writes: models/tokenizer/, eval/tokenizer_compression.md

Run:    python scripts/05_train_tokenizer.py
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    raise NotImplementedError("Phase 5 implementation pending.")


if __name__ == "__main__":
    main()
