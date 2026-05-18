# Eval plan — ryGPT

Detailed methodology for Phase 7. Each section: *what we measure, how, what counts as pass.*

## 1. Perplexity

**What:** Per-token perplexity on the held-out val set, broken down by relationship.

**How:**
- Compute on `data/processed/val.jsonl`.
- Forward pass with the chat template applied exactly as in training; mask everything except `target` tokens; average NLL over target tokens only.
- Run against:
  - Base model (untuned) — baseline number
  - Tuned model (LoRA merged or LoRA-on-base)
- Break out: overall, by relationship (`gf` / `friend` / `group`), by `context_msg_count` bucket.

**Pass:** Tuned ≥ 25% lower than base, no relationship slice worse than base.

**Watch for:** if `group` perplexity is much worse than `gf`/`friend`, the group-chat threading issue is real and we may need to revisit Phase 3 context construction.

## 2. Style classifier

**What:** Can a classifier distinguish *my* writing from *not-mine* Manglish? If yes, does it call the tuned model's generations *mine*?

**Setup:**
- Positive class: my real messages (`<self>` rows from the anonymized data, holdout split from training data).
- Negative class: **multiple** sources, report each separately (see ADR-003):
  - `(A)` r/Kerala posts in Manglish
  - `(B)` Public Telegram channel messages in Manglish
- Length-controlled variant: truncate every example to first 8 tokens before classifying, neutralizes the "Reddit is long, chat is short" confound.

**Classifier:** Logistic regression on TF-IDF (char 2-5 grams). Simple, transparent, fast. If accuracy on the held-out positive/negative split is <70%, the classifier itself is the bottleneck — bump to a small fine-tuned BERT (e.g. `xlm-roberta-base`).

**Evaluation:**
- Generate 50 samples from the tuned model across fixed contexts (drawn from val) per relationship.
- Run each through the classifier.
- Report: % predicted as "me" against (A), against (B), against length-controlled (A), against length-controlled (B).

**Pass:** ≥ 70% positive rate against the length-controlled negative class (the harder metric).

**Caveat to surface in the writeup:** classifier accuracy on the held-out *positive vs negative* split is the ceiling on this metric. If that's e.g. 85%, no generation can be "called me" more often than that. Report the ceiling alongside the metric.

## 3. Generation samples

**What:** Qualitative review.

**How:**
- For each of `<gf>`, `<friend>`, `<group>` (and `<friend_1>` / `<friend_2>` if ADR-001 lands on (B)/(C)):
  - 10 fixed contexts drawn from val (same contexts across runs — set `eval/sample_prompts.json`)
  - Generate from base model and tuned model with same sampling params (temperature 0.8, top_p 0.95, max_new_tokens 80)
- Write side-by-side to `eval/samples.md`.
- Manual annotation by user: for each, mark `sounds_like_me: yes/no/maybe`, optional note on why.

**Pass:** ≥ 60% `yes` on the 30 annotated samples.

**Pre-commit check:** before committing `eval/samples.md`, manually scan for any leaked PII the regex missed.

## 4. Memorization audit (gate)

**What:** Does the model reproduce training data verbatim?

**How:**
- Sample 200 examples from `train.jsonl`.
- For each: feed the `context` to the tuned model, generate one continuation, compute token overlap (Jaccard on token sets, plus longest common subsequence) against the actual `target`.
- Flag any with >80% Jaccard or LCS ratio.

**Gate:** if flagged rate > 5%, **do not** build the demo. Document in `eval/memorization.md` and trigger remediation:
1. Reduce epochs (try 1-2 instead of 3)
2. Increase LoRA dropout (try 0.1)
3. Dedup near-identical training examples (MinHash; remove pairs with Jaccard > 0.8)
4. Retrain, re-audit

**Watch for:** even at <5% flagged, *which* examples are flagged matters more than the rate. If the flagged ones are short common phrases ("ariyilla", "ok da"), that's not memorization — that's a Manglish stopword. Manual review of all flagged before declaring pass.

## 5. Baseline comparison (writeup table)

For the portfolio README, the headline table is:

| Metric                       | Base model | Tuned model | Δ        |
|------------------------------|-----------:|------------:|---------:|
| Val perplexity (overall)     |            |             |          |
| Val perplexity (`<gf>`)      |            |             |          |
| Val perplexity (`<friend>`)  |            |             |          |
| Val perplexity (`<group>`)   |            |             |          |
| Style classifier "me" rate (vs r/Kerala) |  |       |          |
| Style classifier "me" rate (vs Telegram) |  |       |          |
| Style classifier "me" rate (length-controlled) |  |  |          |
| Manual "sounds like me" rate (n=30) |    |             |          |

Plus a memorization-rate line below the table (no Δ — only the tuned model can memorize).

## 6. What we are NOT measuring (and why)

- **BLEU / ROUGE against held-out targets** — chat replies are open-ended; n-gram overlap with one specific held-out reply is noise, not signal.
- **Human preference vs base model in head-to-head** — would need a second annotator to be meaningful; out of scope for a one-person project.
- **Toxicity / safety metrics** — this isn't a deployed assistant; the only audience is me. Generations going off the rails is interesting failure-mode color for the writeup, not a release blocker.
