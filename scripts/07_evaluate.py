"""Phase 7 — perplexity, style classifier, samples, memorization audit.

See .docs/EVAL_PLAN.md for methodology. The memorization audit is a hard gate
on Phase 8 (no demo if >5% of sampled training targets are reproduced).

Reads:  data/processed/{train,val}.jsonl, models/lora_adapter/
Writes: eval/{perplexity,style_classifier,samples,memorization}.md

Run:    python scripts/07_evaluate.py
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/lora_adapter"))
    parser.add_argument("--val-file", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval"))
    parser.add_argument("--memorization-sample-size", type=int, default=200)
    parser.add_argument("--memorization-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raise NotImplementedError("Phase 7 implementation pending.")


if __name__ == "__main__":
    main()
