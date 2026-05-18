"""Phase 3 — segment into sessions, build training examples, split train/val.

Reads:  data/anonymized/*.json
Writes: data/processed/train.jsonl, data/processed/val.jsonl

Relationship tagging:
- --gf-chat <name>  (filename stem of the gf chat; default: 'fay')
- 1 non-self speaker → 'friend'; >1 → 'group'
- Per ADR-001 option C: relationship token is collapsed (<gf> / <friend> / <group>);
  individual <person_N> speakers stay inside the context.

Run:    python scripts/03_build_dataset.py
"""
import _bootstrap  # noqa: F401

import argparse
import json
from collections import Counter
from pathlib import Path

from src.dataset import (
    build_examples_from_chat,
    detect_relationship,
    split_train_val,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", type=Path, default=Path("data/anonymized"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--gf-chat", default="fay",
                        help="Chat stem treated as the gf chat (default: fay)")
    parser.add_argument("--session-gap-hours", type=float, default=2.0,
                        help="see ADR-002")
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.in_dir.exists():
        raise SystemExit(f"Input dir not found: {args.in_dir}. Run Phase 2 first.")

    chat_files = sorted(
        p for p in args.in_dir.glob("*.json") if p.name != "name_mapping.json"
    )
    if not chat_files:
        raise SystemExit(f"No anonymized chat JSONs found in {args.in_dir}")

    all_examples: list[dict] = []
    per_chat_breakdown: list[tuple[str, str, int]] = []

    for p in chat_files:
        with p.open(encoding="utf-8") as f:
            messages = json.load(f)
        chat_name = p.stem
        relationship = detect_relationship(chat_name, messages, gf_chat=args.gf_chat)
        examples = build_examples_from_chat(
            chat_name=chat_name,
            messages=messages,
            relationship=relationship,
            gap_hours=args.session_gap_hours,
            context_size=args.context_size,
        )
        per_chat_breakdown.append((chat_name, relationship, len(examples)))
        all_examples.extend(examples)

    print("Per-chat training examples generated:")
    for chat_name, relationship, n in per_chat_breakdown:
        print(f"  {chat_name:20s} [{relationship:6s}] {n:>7,} examples")
    print(f"  {'total':20s} {'':6s}  {len(all_examples):>7,} examples")
    print()

    train, val = split_train_val(
        all_examples, val_fraction=args.val_fraction, seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    rel_train = Counter(e["relationship"] for e in train)
    rel_val = Counter(e["relationship"] for e in val)
    sessions_train = len({e["session_id"] for e in train})
    sessions_val = len({e["session_id"] for e in val})

    print(f"train: {len(train):>7,} examples across {sessions_train:>5,} sessions -> {train_path}")
    print(f"       by relationship: {dict(rel_train)}")
    print(f"val:   {len(val):>7,} examples across {sessions_val:>5,} sessions -> {val_path}")
    print(f"       by relationship: {dict(rel_val)}")


if __name__ == "__main__":
    main()
