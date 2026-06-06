# Decisions log — ryGPT

Lightweight ADR-style log. Each entry: *what was decided, why, what alternatives were rejected, what would force a re-decision*. Open items at top.

---

## Open — to decide before relevant phase starts

### ADR-001 — Relationship token granularity (decide before Phase 3)
**Question:** Use a single `<friend>` token for both individual-friend chats, or distinct `<friend_1>` / `<friend_2>`?

**Options:**
- **(A) Collapse to `<friend>`** — matches PROJECT.md literal spec, fewer tokens, more data per token.
- **(B) Distinct `<friend_1>` / `<friend_2>`** — preserves per-person register signal. Anonymization still holds (the token doesn't identify the friend to anyone reading the model).
- **(C) Collapse but keep `<person_N>` speaker IDs in context** — relationship token says "friend," but the context still shows which `<person_N>` is talking. The model can learn person-specific style from context alone, without it leaking into the top-level conditioning.

**Currently leaning toward (C)** — best of both, doesn't multiply rare tokens, preserves signal that's already in `context[].speaker`.

**Forces re-decision:** if Phase 4 stats show per-friend message counts are very lopsided, may want (B) so the underrepresented friend doesn't get drowned.

### ADR-002 — Session gap threshold (decide before Phase 3)
**Question:** PROJECT.md says >2h. Reasonable?

**Default:** 2h.
**Concern:** WhatsApp chat often resumes after 6–12h with continuous topic. Cutting at 2h fragments coherent conversations and inflates session count.
**Plan:** Implement as a CLI arg with default 2h. In Phase 4 stats, print the session-count distribution at multiple thresholds (1h, 2h, 4h, 6h, 8h). Pick the elbow.

**Forces re-decision:** if 2h produces >10× more sessions than 6h with similar avg length, the longer threshold is probably right.

### ADR-003 — Style classifier negative class (decide before Phase 7)
**Question:** What's the "not-me" data for the style classifier?

**PROJECT.md suggests:** scraped Manglish from r/Kerala.

**Problem:** r/Kerala is long-form, political/cultural posts. My chats are short, conversational, code-switched. Classifier may learn "short vs long" not "me vs not-me."

**Options:**
- **(A) r/Kerala only** — fastest, but confounded.
- **(B) Public Telegram Manglish channels** (closer to chat register).
- **(C) Multiple negative classes, report each separately** (recommended).
- **(D) Self-vs-self holdout** — train classifier on (me-train, not-me) where "not-me" is a friend's messages from the same chats. Cleanest but anonymized data complicates this — `<self>` is the only label, friends' lines lose their identity.

**Currently leaning toward (C) with (A) + (B)** — report both numbers. Plus a "length-controlled" classifier that only sees the first 8 tokens to neutralize the length confound.

**Forces re-decision:** if accuracy varies wildly across (A) and (B), domain confound is real and the eval needs to be reported with that caveat.

### ADR-004 — Base model (decide after Phase 5)
**Question:** Qwen2.5-1.5B vs Llama-3.2-1B.

**Plan:** Run a quick perplexity check on a held-out anonymized sample with each. Pick the lower-perplexity base. If they're within 5%, prefer Qwen2.5-1.5B (better multilingual baseline historically).

**Forces re-decision:** GPU memory headroom on the rented box, or licensing constraints from Llama if the model is ever shared.

---

## Decided

### ADR-009 — Class imbalance across relationships (decided 2026-05-18)
**Decision:** Train on the full dataset as-is. No downsampling, no stratification, no per-relationship loss weighting. Accept that ~97% of messages are from the Fay chat (`<gf>`).

**Why:** The user's intended primary use of the model is gf-style generation. Equalizing relationships would dilute the target style. Per-relationship eval numbers will show `<gf>` strongest; that's intentional bias, not a failure.

**Forces re-decision:** if Phase 7 shows `<friend>` / `<group>` perplexity is *worse than the untuned base* (i.e. negative transfer from gf-dominant data), revisit.

### ADR-005 — Custom tokenizer extension threshold
**Decision:** Extend base vocab only if custom BPE compresses Manglish ≥ 1.5× better than base tokenizer.

**Why:** Below 1.5×, the embedding-resize + retraining cost isn't worth the marginal compression. Above 1.5×, longer sequences are eating real budget.

**Source:** PROJECT.md Phase 5.

**Outcome (2026-05-19):** Custom BPE measured **1.41×** on targets, **1.28×** on full sequences. Below threshold → **skipped**. Base Qwen2.5 tokenizer used as-is.

### ADR-010 — Don't add `<self>` / `<gf>` / `<person_N>` / `[media]` etc. as atomic tokens
**Decision:** Let the base Qwen2.5 tokenizer fragment our project tokens naturally (`<self>` → 3 subword pieces, `<person_N>` → 5). Do NOT call `tokenizer.add_special_tokens` or `model.resize_token_embeddings` anywhere in the pipeline.

**Why:** First smoke run (1000 examples, 1 epoch) produced **Thai script** at every speaker prefix. Root cause: adding 15 new tokens inserted random rows into the embedding + lm_head matrices. The LoRA config only trains attention/MLP — it does NOT train the embedding or lm_head. So the new rows stayed randomly initialized and the model emitted random nearby tokens at those positions.

The alternatives were (a) `modules_to_save=["embed_tokens", "lm_head"]` which roughly doubles VRAM and was risky on a 24 GB box, or (b) smart embedding init (mean-of-existing) which still leaves the rows mostly stale. Fragmenting is the simplest path: the model just learns the multi-token sequence pattern from repetition (every example has `<self>:` at the start of the target).

**How to apply:** In `scripts/06_train_model.py`, `07_evaluate.py`, `08_demo.py`, `chat.py`: do NOT modify the tokenizer or model vocab size. The chat-template formatter in `src/dataset.py` puts the tokens in as plain text and that's correct.

**Forces re-decision:** if Phase 7 generations show the model consistently failing to emit the speaker prefix (rather than learning the pattern), revisit by trying `modules_to_save` with reduced batch size.

### ADR-006 — Memorization gate on demo release
**Decision:** No demo (even local Gradio) if >5% of sampled training targets are reproduced with >80% token overlap in generations.

**Why:** Memorization at this corpus size is plausible. Personal chat content reproduced verbatim is the worst-case failure for this project.

**Mitigations if gate trips:** more regularization (higher LoRA dropout), fewer epochs, more aggressive dedup of near-duplicate training examples.

### ADR-007 — Train/val split granularity
**Decision:** Split by `session_id`, not by message.

**Why:** Messages within a session are conditionally dependent. Splitting by message leaks future context into val and inflates eval numbers.

**Source:** PROJECT.md Phase 3.

### ADR-008 — Privacy posture
**Decision:**
- Raw chat data and `name_mapping.json` never enter version control.
- Anonymized derivatives also gitignored (still personal content, just de-identified).
- Model not released publicly without explicit decision + passed memorization audit.
- Eval markdown files in `eval/` may contain generated samples; review before commit.

**Why:** This is non-public personal data. The audit cost of accidentally committing it is real. Default-deny via `.gitignore`.

---

## Template for new entries
```
### ADR-NNN — short title
**Question / Decision:** ...
**Why:** ...
**Alternatives:** ...
**Forces re-decision:** ...
```
