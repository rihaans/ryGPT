"""Phase 4 — data stats + manual anonymization review samples.

STOP point per PROJECT.md: review the outputs before kicking off Phase 5.

Reads:  data/anonymized/*.json, data/processed/{train,val}.jsonl
Writes: eval/data_stats.md, eval/data_stats.json,
        eval/anon_review_samples.txt   (gitignored — may contain PII)

Run:    python scripts/04_data_stats.py
"""
import _bootstrap  # noqa: F401

import argparse
import json
import random
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.dataset import (
    detect_relationship,
    read_jsonl,
    segment_into_sessions,
)


def _percentiles(xs: list[int], qs: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> dict:
    if not xs:
        return {q: 0 for q in qs}
    xs_sorted = sorted(xs)
    out = {}
    for q in qs:
        i = min(len(xs_sorted) - 1, int(q * len(xs_sorted)))
        out[q] = xs_sorted[i]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--anonymized-dir", type=Path, default=Path("data/anonymized"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval"))
    parser.add_argument("--gf-chat", default="fay")
    parser.add_argument("--review-sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.anonymized_dir.exists():
        raise SystemExit(f"Anonymized dir not found: {args.anonymized_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_lines: list[str] = []
    machine: dict = {}

    md_lines.append("# Phase 4 — Data stats")
    md_lines.append("")
    md_lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    md_lines.append("")

    # ---- Per-chat (from anonymized JSONs) ----
    chat_files = sorted(
        p for p in args.anonymized_dir.glob("*.json") if p.name != "name_mapping.json"
    )
    per_chat: list[dict] = []
    grand_total_msgs = 0
    grand_total_self = 0

    for p in chat_files:
        with p.open(encoding="utf-8") as f:
            msgs = json.load(f)
        rel = detect_relationship(p.stem, msgs, gf_chat=args.gf_chat)
        n = len(msgs)
        n_self = sum(1 for m in msgs if m["speaker"] == "<self>")
        media = sum(1 for m in msgs if m["text"] == "[media]")
        # Session counts at multiple thresholds (see ADR-002)
        gap_session_counts = {
            f"{h}h": len(segment_into_sessions(msgs, gap_hours=float(h)))
            for h in (1, 2, 4, 6, 8, 12, 24)
        }
        per_chat.append({
            "chat": p.stem,
            "relationship": rel,
            "messages": n,
            "self_messages": n_self,
            "media_count": media,
            "sessions_at_gap": gap_session_counts,
        })
        grand_total_msgs += n
        grand_total_self += n_self

    md_lines.append("## Overall")
    md_lines.append(f"- Total anonymized messages: **{grand_total_msgs:,}**")
    md_lines.append(f"- Of which `<self>`: **{grand_total_self:,}** "
                    f"({grand_total_self / grand_total_msgs:.1%})")
    md_lines.append("")
    machine["overall"] = {
        "messages": grand_total_msgs,
        "self_messages": grand_total_self,
    }

    md_lines.append("## Per-chat")
    md_lines.append("")
    md_lines.append("| chat | relationship | messages | `<self>` | `[media]` | self share |")
    md_lines.append("|------|--------------|---------:|--------:|---------:|-----------:|")
    for c in per_chat:
        share = c["self_messages"] / c["messages"] if c["messages"] else 0
        md_lines.append(
            f"| {c['chat']} | {c['relationship']} | "
            f"{c['messages']:,} | {c['self_messages']:,} | {c['media_count']:,} | "
            f"{share:.1%} |"
        )
    md_lines.append("")
    machine["per_chat"] = per_chat

    # ---- Session count sensitivity to gap threshold ----
    md_lines.append("## Session segmentation sensitivity (ADR-002)")
    md_lines.append("")
    md_lines.append("Session count per chat at different gap thresholds.")
    md_lines.append("")
    md_lines.append("| chat | 1h | 2h | 4h | 6h | 8h | 12h | 24h |")
    md_lines.append("|------|---:|---:|---:|---:|---:|----:|----:|")
    for c in per_chat:
        s = c["sessions_at_gap"]
        md_lines.append(
            f"| {c['chat']} | {s['1h']:,} | {s['2h']:,} | {s['4h']:,} | "
            f"{s['6h']:,} | {s['8h']:,} | {s['12h']:,} | {s['24h']:,} |"
        )
    md_lines.append("")

    # ---- Training examples (from train.jsonl / val.jsonl) ----
    train_path = args.processed_dir / "train.jsonl"
    val_path = args.processed_dir / "val.jsonl"
    if train_path.exists() and val_path.exists():
        train = read_jsonl(train_path)
        val = read_jsonl(val_path)
        n_train = len(train)
        n_val = len(val)
        train_sessions = len({e["session_id"] for e in train})
        val_sessions = len({e["session_id"] for e in val})

        rel_train = Counter(e["relationship"] for e in train)
        rel_val = Counter(e["relationship"] for e in val)

        # Target length distribution (chars and words)
        target_chars = [len(e["target"]) for e in train]
        target_words = [len(e["target"].split()) for e in train]
        ctx_msgs = [e["context_msg_count"] for e in train]

        md_lines.append("## Training examples")
        md_lines.append("")
        md_lines.append(f"- Train: **{n_train:,}** examples across **{train_sessions:,}** sessions")
        md_lines.append(f"- Val:   **{n_val:,}** examples across **{val_sessions:,}** sessions")
        md_lines.append(
            f"- Train relationship mix: {dict(rel_train)}"
        )
        md_lines.append(
            f"- Val relationship mix:   {dict(rel_val)}"
        )
        md_lines.append("")

        def _pctline(label: str, xs: list[int]) -> str:
            if not xs:
                return f"| {label} | 0 | 0 | 0 | 0 | 0 |"
            p = _percentiles(xs)
            return (f"| {label} | {statistics.mean(xs):.1f} | "
                    f"{p[0.5]} | {p[0.9]} | {p[0.95]} | {p[0.99]} |")

        md_lines.append("### Length distributions (train set)")
        md_lines.append("")
        md_lines.append("| metric | mean | p50 | p90 | p95 | p99 |")
        md_lines.append("|--------|-----:|----:|----:|----:|----:|")
        md_lines.append(_pctline("target chars", target_chars))
        md_lines.append(_pctline("target words", target_words))
        md_lines.append(_pctline("context msgs", ctx_msgs))
        md_lines.append("")

        machine["training"] = {
            "train_examples": n_train, "val_examples": n_val,
            "train_sessions": train_sessions, "val_sessions": val_sessions,
            "rel_train": dict(rel_train), "rel_val": dict(rel_val),
            "target_chars": {"mean": statistics.mean(target_chars),
                             **{f"p{int(q*100)}": v for q, v in _percentiles(target_chars).items()}},
            "target_words": {"mean": statistics.mean(target_words),
                             **{f"p{int(q*100)}": v for q, v in _percentiles(target_words).items()}},
            "context_msg_count": {"mean": statistics.mean(ctx_msgs),
                                   **{f"p{int(q*100)}": v for q, v in _percentiles(ctx_msgs).items()}},
        }

        # ---- Anonymization review samples ----
        rng = random.Random(args.seed)
        review_pool = [e for e in train if e["target"] != "[media]"]
        sample = rng.sample(review_pool, k=min(args.review_sample_size, len(review_pool)))
        review_path = args.out_dir / "anon_review_samples.txt"
        with review_path.open("w", encoding="utf-8") as f:
            f.write("# Manual anonymization review\n")
            f.write(f"# {len(sample)} random training targets — scan for any leaked PII\n")
            f.write("# (real names, addresses, phone numbers, etc.) that the regex missed.\n")
            f.write("# This file is gitignored. Do NOT commit.\n\n")
            for i, ex in enumerate(sample, 1):
                f.write(f"--- sample {i} | session {ex['session_id']} | "
                        f"{ex['relationship']} | ctx={ex['context_msg_count']} ---\n")
                for c in ex["context"]:
                    f.write(f"  {c['speaker']}: {c['text']}\n")
                f.write(f"  >>> <self>: {ex['target']}\n\n")
        md_lines.append(f"## Manual review samples")
        md_lines.append("")
        md_lines.append(f"{len(sample)} random training examples written to "
                        f"`{review_path}` (gitignored — do not commit).")
        md_lines.append("")
        md_lines.append("**Action:** open the file and scan for any leaked PII the regex missed "
                        "(real names not in the mapping, addresses, family member names, "
                        "workplace/school names, etc.). If you find any, add them as aliases "
                        "in `data/anonymized/name_mapping.json` and re-run Phase 2 + 3 + 4.")
        md_lines.append("")
    else:
        md_lines.append("## Training examples")
        md_lines.append("")
        md_lines.append("_(train.jsonl / val.jsonl not found — run Phase 3 first.)_")
        md_lines.append("")

    # ---- Write outputs ----
    md_path = args.out_dir / "data_stats.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    json_path = args.out_dir / "data_stats.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    if train_path.exists():
        print(f"Wrote {args.out_dir / 'anon_review_samples.txt'} (gitignored)")
    print()
    print("STOP. Open the markdown, scan the anonymization review samples,")
    print("and confirm before proceeding to Phase 5.")


if __name__ == "__main__":
    main()
