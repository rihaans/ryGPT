"""Phase 3 — segment into sessions, build training examples, split train/val.

Reads:  data/anonymized/*.json
Writes: data/processed/train.jsonl, data/processed/val.jsonl

Run:    python scripts/03_build_dataset.py
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", type=Path, default=Path("data/anonymized"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--session-gap-hours", type=float, default=2.0, help="see ADR-002")
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raise NotImplementedError("Phase 3 implementation pending.")


if __name__ == "__main__":
    main()
