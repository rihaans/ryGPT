"""Phase 5 — train a custom BPE tokenizer, compare compression vs base.

STOP point. Decision: extend the base model's vocab with custom tokens only if
the custom BPE compresses Manglish ≥ 1.5x better than the base tokenizer (ADR-005).

Reads:  data/processed/{train,val}.jsonl, data/anonymized/name_mapping.json
Writes: models/tokenizer/tokenizer.json,
        eval/tokenizer_compression.md
"""
import _bootstrap  # noqa: F401

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer

from src.dataset import read_jsonl


def _corpus_iter(train_examples: list[dict]) -> Iterator[str]:
    """Stream every distinct text string (target + context messages) through the trainer."""
    seen: set[str] = set()
    for ex in train_examples:
        for s in (ex["target"], *(c["text"] for c in ex["context"])):
            if s and s not in seen:
                seen.add(s)
                yield s


def _build_special_tokens(name_mapping_path: Path) -> list[str]:
    """All speaker tokens, relationship tokens, and scrub placeholders."""
    specials = ["<self>", "<gf>", "<friend>", "<group>",
                "[media]", "[phone]", "[email]", "[upi]", "[number]", "[link]"]
    if name_mapping_path.exists():
        with name_mapping_path.open(encoding="utf-8") as f:
            mapping = json.load(f)
        for tok in mapping:
            if tok.startswith("<person_") and tok not in specials:
                specials.append(tok)
    return specials


def train_custom_tokenizer(
    train_examples: list[dict],
    vocab_size: int,
    specials: list[str],
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", *specials],
        show_progress=True,
    )
    tokenizer.train_from_iterator(_corpus_iter(train_examples), trainer)
    return tokenizer


def _count_tokens_custom(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False).ids)


def _count_tokens_base(tok, text: str) -> int:
    # transformers tokenizer
    return len(tok.encode(text, add_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--name-mapping", type=Path,
                        default=Path("data/anonymized/name_mapping.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/tokenizer"))
    parser.add_argument("--eval-out", type=Path,
                        default=Path("eval/tokenizer_compression.md"))
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--compare-sample", type=int, default=5_000,
                        help="Number of val targets to measure compression on")
    args = parser.parse_args()

    if not args.train_file.exists():
        raise SystemExit(f"Train file not found: {args.train_file}. Run Phase 3 first.")

    print(f"Loading {args.train_file} …")
    train = read_jsonl(args.train_file)
    val = read_jsonl(args.val_file) if args.val_file.exists() else []
    print(f"  train: {len(train):,} examples; val: {len(val):,} examples")

    specials = _build_special_tokens(args.name_mapping)
    print(f"  special tokens ({len(specials)}): {specials}")

    print(f"\nTraining custom BPE tokenizer (vocab={args.vocab_size:,}) …")
    custom = train_custom_tokenizer(train, args.vocab_size, specials)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "tokenizer.json"
    custom.save(str(out_path))
    print(f"  saved -> {out_path}")

    print(f"\nLoading base tokenizer: {args.base_model} …")
    base = AutoTokenizer.from_pretrained(args.base_model)

    sample = val[: args.compare_sample] if val else train[: args.compare_sample]
    sample = [ex for ex in sample if ex["target"] and ex["target"] != "[media]"]
    print(f"  comparing on {len(sample)} target strings")

    base_counts = []
    custom_counts = []
    for ex in sample:
        t = ex["target"]
        base_counts.append(_count_tokens_base(base, t))
        custom_counts.append(_count_tokens_custom(custom, t))

    total_base = sum(base_counts)
    total_custom = sum(custom_counts)
    avg_base = statistics.mean(base_counts)
    avg_custom = statistics.mean(custom_counts)
    ratio = total_base / max(total_custom, 1)

    # Also measure on the full context+target (a more realistic per-example sequence length).
    full_base_counts = []
    full_custom_counts = []
    for ex in sample:
        joined = ex["target"] + " " + " ".join(c["text"] for c in ex["context"])
        full_base_counts.append(_count_tokens_base(base, joined))
        full_custom_counts.append(_count_tokens_custom(custom, joined))
    full_ratio = sum(full_base_counts) / max(sum(full_custom_counts), 1)

    # Custom vocab is fully Manglish-trained, so its tokens-per-message should reach a floor.
    # Smaller numbers = better compression.
    decision = "extend (custom ≥ 1.5x base)" if ratio >= 1.5 else "skip (custom < 1.5x base)"

    md = [
        "# Phase 5 — Tokenizer compression",
        "",
        f"- Base model: `{args.base_model}`",
        f"- Custom BPE vocab size: {args.vocab_size:,}",
        f"- Sample: {len(sample):,} held-out target strings",
        "",
        "## Target-only compression",
        "",
        "| tokenizer | total tokens | mean / target |",
        "|-----------|-------------:|--------------:|",
        f"| base (`{args.base_model}`) | {total_base:,} | {avg_base:.2f} |",
        f"| custom 16k BPE | {total_custom:,} | {avg_custom:.2f} |",
        f"| **ratio** | **{ratio:.2f}x** | — |",
        "",
        "## Full sequence (context + target) compression",
        "",
        "| tokenizer | total tokens | mean / example |",
        "|-----------|-------------:|---------------:|",
        f"| base | {sum(full_base_counts):,} | "
        f"{statistics.mean(full_base_counts):.2f} |",
        f"| custom | {sum(full_custom_counts):,} | "
        f"{statistics.mean(full_custom_counts):.2f} |",
        f"| **ratio** | **{full_ratio:.2f}x** | — |",
        "",
        "## Decision",
        "",
        f"Threshold per ADR-005: extend if target-only ratio ≥ 1.5x.  ",
        f"Result: **{decision}**",
        "",
        "## Notes",
        "",
        "- Ratio < 1 means base tokenizer is more efficient. This can happen when "
        "the base tokenizer already merges common Manglish subwords (Qwen2.5 has "
        "broad multilingual coverage).",
        "- Even if we skip vocab extension, the relationship/speaker/scrub special "
        "tokens are still added to the base tokenizer as **added tokens** during "
        "Phase 6, so the model treats them as atomic units.",
        "",
    ]
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    args.eval_out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.eval_out}")

    print()
    print(f"Target-only ratio: {ratio:.2f}x ({decision})")
    print(f"Full-sequence ratio: {full_ratio:.2f}x")
    print()
    print("STOP. Open the markdown and confirm the decision before Phase 6.")


if __name__ == "__main__":
    main()
