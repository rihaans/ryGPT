"""Phase 7 — perplexity, generation samples, style classifier, memorization audit.

Run on the GPU box (same env as training). See `.docs/EVAL_PLAN.md` for the methodology
and pass criteria.

Reads:
  data/processed/{train,val}.jsonl
  models/lora_adapter/  (output of Phase 6)
  data/eval_negatives/*.txt  (one negative-class corpus per file, optional)

Writes:
  eval/perplexity.md
  eval/samples.md
  eval/style_classifier.md   (only if negative-class data is present)
  eval/memorization.md

Run:
    python scripts/07_evaluate.py
"""
import _bootstrap  # noqa: F401

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", type=str, default=None,
                        help="Override; defaults to value from training_config.json")
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/lora_adapter"))
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--negatives-dir", type=Path, default=Path("data/eval_negatives"),
                        help="Directory of .txt files, one negative-class corpus each "
                             "(e.g. reddit_kerala.txt, telegram_manglish.txt)")
    parser.add_argument("--out-dir", type=Path, default=Path("eval"))
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--ppl-sample", type=int, default=2000,
                        help="Cap on val examples for perplexity (use 0 for full val)")
    parser.add_argument("--gen-per-relationship", type=int, default=10)
    parser.add_argument("--gen-temperature", type=float, default=0.8)
    parser.add_argument("--gen-top-p", type=float, default=0.95)
    parser.add_argument("--gen-max-new-tokens", type=int, default=80)
    parser.add_argument("--memorization-sample", type=int, default=200)
    parser.add_argument("--memorization-jaccard-threshold", type=float, default=0.8)
    parser.add_argument("--memorization-lcs-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            f"Missing deps ({e.name}). Install training stack on the eval box first."
        )

    from src.dataset import read_jsonl
    from src.eval import (
        classify_as_me,
        compute_perplexity,
        generate_response,
        memorization_flag,
        train_style_classifier,
        truncate_to_tokens,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- Base model name comes from training_config.json by default ----
    cfg_path = args.adapter_dir / "training_config.json"
    if args.base_model is None:
        if not cfg_path.exists():
            raise SystemExit(
                f"--base-model not given and {cfg_path} not found. "
                "Specify --base-model explicitly."
            )
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        args.base_model = cfg["base_model"]
    print(f"Base model: {args.base_model}")
    print(f"Adapter:    {args.adapter_dir}")

    # ---- Load tokenizer + base + tuned (LoRA-on-base) ----
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model …")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base.resize_token_embeddings(len(tokenizer))

    print("Loading base model (separate copy) for adapter attach …")
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base_for_lora.resize_token_embeddings(len(tokenizer))
    tuned = PeftModel.from_pretrained(base_for_lora, args.adapter_dir)
    tuned.eval()
    base.eval()

    # ---- Data ----
    val = read_jsonl(args.val_file)
    train = read_jsonl(args.train_file)
    if args.ppl_sample > 0 and len(val) > args.ppl_sample:
        ppl_subset = rng.sample(val, args.ppl_sample)
    else:
        ppl_subset = val

    # ---- 1. Perplexity (base vs tuned) ----
    print(f"\nPerplexity on {len(ppl_subset):,} val examples …")
    ppl_base = compute_perplexity(
        base, tokenizer, ppl_subset, max_seq_length=args.max_seq_length,
    )
    ppl_tuned = compute_perplexity(
        tuned, tokenizer, ppl_subset, max_seq_length=args.max_seq_length,
    )

    ppl_md = [
        "# Phase 7 — Perplexity",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        f"_Sample size: {len(ppl_subset):,} val examples_",
        "",
        "| slice | base | tuned | Δ |",
        "|-------|-----:|------:|--:|",
    ]
    for slice_name in ["overall"] + sorted([k for k in ppl_tuned if k != "overall"]):
        b = ppl_base.get(slice_name, {}).get("perplexity", float("nan"))
        t = ppl_tuned.get(slice_name, {}).get("perplexity", float("nan"))
        delta = (t - b) / b * 100 if b and not math.isnan(b) else float("nan")
        ppl_md.append(f"| {slice_name} | {b:.2f} | {t:.2f} | {delta:+.1f}% |")
    ppl_md += [
        "",
        "_Pass criterion (EVAL_PLAN.md): tuned ≥ 25% lower than base on overall, "
        "no relationship slice worse than base._",
        "",
    ]
    (args.out_dir / "perplexity.md").write_text("\n".join(ppl_md), encoding="utf-8")
    print("  -> eval/perplexity.md")

    # ---- 2. Generation samples (base vs tuned, side by side) ----
    print(f"\nGeneration samples …")
    # Pick fixed contexts per relationship from val.
    samples_md = [
        "# Phase 7 — Generation samples",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        f"_temperature={args.gen_temperature}, top_p={args.gen_top_p}, "
        f"max_new_tokens={args.gen_max_new_tokens}_",
        "",
        "_For each fixed context, BASE = untuned model, TUNED = with LoRA adapter._",
        "_Manual annotation: mark each TUNED row with `me: yes/no/maybe`._",
        "",
    ]
    relationships = sorted({e["relationship"] for e in val})
    chosen_examples = []
    for rel in relationships:
        rel_pool = [e for e in val if e["relationship"] == rel]
        n = min(args.gen_per_relationship, len(rel_pool))
        chosen = rng.sample(rel_pool, n)
        chosen_examples.extend((rel, ex) for ex in chosen)

    for rel in relationships:
        samples_md.append(f"## {rel}")
        samples_md.append("")
        for i, (r, ex) in enumerate([t for t in chosen_examples if t[0] == rel], 1):
            base_gen = generate_response(
                base, tokenizer, ex,
                max_new_tokens=args.gen_max_new_tokens,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                seed=args.seed + i,
            )
            tuned_gen = generate_response(
                tuned, tokenizer, ex,
                max_new_tokens=args.gen_max_new_tokens,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                seed=args.seed + i,
            )
            samples_md.append(f"### {rel}/{i}")
            samples_md.append("")
            samples_md.append("**Context:**")
            for c in ex["context"]:
                samples_md.append(f"- {c['speaker']}: {c['text']}")
            samples_md.append("")
            samples_md.append(f"**Ground truth target:** {ex['target']}")
            samples_md.append("")
            samples_md.append(f"**BASE  →** {base_gen}")
            samples_md.append("")
            samples_md.append(f"**TUNED →** {tuned_gen}")
            samples_md.append("")
            samples_md.append("`me: ` (yes/no/maybe — fill in manually)")
            samples_md.append("")
    (args.out_dir / "samples.md").write_text("\n".join(samples_md), encoding="utf-8")
    print("  -> eval/samples.md")

    # ---- 3. Style classifier (only if negative-class data is available) ----
    classifier_md = ["# Phase 7 — Style classifier", "",
                     f"_Generated: {datetime.now().isoformat(timespec='seconds')}_", ""]
    neg_files = sorted(args.negatives_dir.glob("*.txt")) if args.negatives_dir.exists() else []
    if not neg_files:
        classifier_md.append(
            f"_No negative-class corpora found in `{args.negatives_dir}`. "
            f"Skipping classifier eval. Drop `.txt` files (one Manglish snippet per line) "
            f"into that directory and re-run — recommended sources: r/Kerala posts, "
            f"public Telegram Manglish channels._"
        )
    else:
        # Positive class: self messages from train (not val) to avoid leakage.
        positive_texts = [
            ex["target"] for ex in train if ex["target"] and ex["target"] != "[media]"
        ]
        # Use a manageable subset.
        rng.shuffle(positive_texts)
        positive_texts = positive_texts[:5000]

        # Generate tuned outputs to score.
        gen_pool = [e for _, e in chosen_examples]
        tuned_generations = [
            generate_response(
                tuned, tokenizer, ex,
                max_new_tokens=args.gen_max_new_tokens,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                seed=args.seed + i,
            )
            for i, ex in enumerate(gen_pool, 1)
        ]

        classifier_md.append("| negative class | holdout accuracy | tuned 'me' rate "
                              "| tuned 'me' rate (length-controlled, 8 tokens) |")
        classifier_md.append("|----------------|-----------------:|"
                              "----------------:|---------------------------:|")
        for nf in neg_files:
            with nf.open(encoding="utf-8") as f:
                neg_texts = [line.strip() for line in f if line.strip()]
            if len(neg_texts) < 50:
                classifier_md.append(
                    f"| {nf.stem} | _too few samples ({len(neg_texts)})_ | — | — |"
                )
                continue
            clf, vec, acc = train_style_classifier(positive_texts, neg_texts)
            me_rate = classify_as_me(clf, vec, tuned_generations)
            # Length-controlled: truncate everything to 8 tokens and retrain.
            pos_lc = [truncate_to_tokens(t, 8) for t in positive_texts]
            neg_lc = [truncate_to_tokens(t, 8) for t in neg_texts]
            clf_lc, vec_lc, _ = train_style_classifier(pos_lc, neg_lc)
            me_rate_lc = classify_as_me(
                clf_lc, vec_lc, [truncate_to_tokens(g, 8) for g in tuned_generations],
            )
            classifier_md.append(
                f"| {nf.stem} | {acc:.2%} | {me_rate:.2%} | {me_rate_lc:.2%} |"
            )
        classifier_md.append("")
        classifier_md.append(
            "_Pass criterion (EVAL_PLAN.md): ≥ 70% positive rate against the "
            "length-controlled negative class._"
        )
    (args.out_dir / "style_classifier.md").write_text(
        "\n".join(classifier_md), encoding="utf-8",
    )
    print("  -> eval/style_classifier.md")

    # ---- 4. Memorization audit ----
    print(f"\nMemorization audit on {args.memorization_sample} train examples …")
    mem_subset = rng.sample(train, min(args.memorization_sample, len(train)))
    flagged: list[dict] = []
    for i, ex in enumerate(mem_subset, 1):
        gen = generate_response(
            tuned, tokenizer, ex,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=0.0 if False else args.gen_temperature,  # keep stochastic per plan
            top_p=args.gen_top_p,
            seed=args.seed + i,
        )
        flag = memorization_flag(
            ex["target"], gen,
            jaccard_threshold=args.memorization_jaccard_threshold,
            lcs_threshold=args.memorization_lcs_threshold,
        )
        if flag["flagged"]:
            flagged.append({
                "session_id": ex["session_id"],
                "relationship": ex["relationship"],
                "target": ex["target"],
                "generation": gen,
                **flag,
            })

    flagged_rate = len(flagged) / len(mem_subset) if mem_subset else 0
    gate_passes = flagged_rate <= 0.05

    mem_md = [
        "# Phase 7 — Memorization audit",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        f"_Sample: {len(mem_subset)} train examples_",
        f"_Thresholds: Jaccard ≥ {args.memorization_jaccard_threshold}, "
        f"LCS ≥ {args.memorization_lcs_threshold}_",
        "",
        f"**Flagged: {len(flagged)} / {len(mem_subset)} = {flagged_rate:.1%}**",
        "",
        f"**Phase 8 gate: {'PASSED — demo OK' if gate_passes else 'FAILED — do not deploy demo'}**",
        "",
        "_Manual review note: short common Manglish phrases (e.g. 'ariyilla', 'ok da') "
        "may flag without being true memorization. Inspect flagged examples below "
        "before declaring failure._",
        "",
        "## Flagged examples",
        "",
    ]
    for i, f in enumerate(flagged, 1):
        mem_md.append(f"### {i}. {f['relationship']} | jaccard={f['jaccard']:.2f} "
                       f"| lcs={f['lcs']:.2f}")
        mem_md.append(f"- target: {f['target']}")
        mem_md.append(f"- gen:    {f['generation']}")
        mem_md.append("")
    (args.out_dir / "memorization.md").write_text("\n".join(mem_md), encoding="utf-8")
    print(f"  flagged rate: {flagged_rate:.1%} ({len(flagged)}/{len(mem_subset)})")
    print(f"  gate: {'PASSED' if gate_passes else 'FAILED'}")
    print("  -> eval/memorization.md")

    print("\nDone. Review eval/*.md before launching Phase 8 (demo).")


if __name__ == "__main__":
    main()
