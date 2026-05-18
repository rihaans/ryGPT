# PRD — ryGPT (Manglish Personal LM)

## Problem
Most "personal AI" demos use prompt-only personas. They don't actually capture how a specific person writes — code-switching, slang, in-jokes, register shifts between people. I want a model that produces Manglish (Malayalam in Latin script) in *my* personal style, conditioned on who I'm talking to.

## Users
Single user: the project author. Not a product. Output is one fine-tuned model checkpoint plus a portfolio writeup.

## Goals (in priority order)
1. **A model that writes in my voice** — generations are stylistically distinguishable from generic Manglish (Reddit r/Kerala, public Telegram, etc.) on a held-out classifier eval.
2. **Honest, reproducible methodology** — every claim in the writeup is backed by a number from `eval/` outputs; no cherry-picked samples.
3. **Privacy-clean** — no chat content, no name mapping, no media in version control; ever. Model not released publicly without a passed memorization audit.
4. **Portfolio-grade writeup** — README walks a reader through problem → data → method → results → failure modes, with numbers and one or two illustrative samples.

## Non-goals
- Multi-modal (no images, voice, stickers)
- RLHF / preference tuning
- Agentic / tool-using behavior
- Public deployment
- Multi-language support beyond Manglish + some English code-switching that's already in my chats
- Beating any benchmark — this is a personal-style project, not a leaderboard chase

## Success criteria
| # | Metric | Target | Source |
|---|--------|--------|--------|
| 1 | Val perplexity vs. base model (same val set) | Tuned ≥ 25% lower than base | Phase 7 |
| 2 | Style classifier: does it call tuned outputs "me"? | ≥ 70% positive rate on 50 generations | Phase 7 |
| 3 | Memorization audit: training examples with >80% token overlap in generation | < 5% of sampled training set | Phase 7 |
| 4 | Manual eval: "does this sound like me?" on 30 blind samples | Subjective ≥ 60% pass | Phase 7 |

If #3 fails (>5% memorization), the model is not released even locally as a demo — back to data dedup / regularization.

## Constraints
- **Data is small** (4 chats, exact volume TBD in Phase 4) → overfitting is the dominant risk, drives QLoRA + 1.5B base + early stopping.
- **No labeled style data** for Manglish — eval has to be self-constructed (see [EVAL_PLAN.md](EVAL_PLAN.md) for the negative-class problem).
- **Single rented GPU budget** — keep training runs under ~4h, keep the total number of runs in single digits.
- **Local-only data** — raw exports never leave the dev machine; only anonymized derivatives ever sit on rented hardware, and only for the duration of a training run.

## Risks
1. **Manglish orthographic noise** (same word spelled many ways) → tokenizer fragmentation, low effective n-gram overlap. *Mitigation:* Phase 5 measures this directly.
2. **Group chat threading** — replies aren't always to the most recent message. *Mitigation:* Phase 4 stats flag how often this occurs; may need group-specific context construction.
3. **Style classifier domain confound** — generic Manglish from public sources is long-form while my chats are short. Classifier may learn length rather than style. *Mitigation:* enumerate negative-class options in [EVAL_PLAN.md](EVAL_PLAN.md), report multiple.
4. **Memorization** at this corpus size is plausible. *Mitigation:* explicit audit in Phase 7 with a hard gate on release.

## Out-of-scope decisions deferred to ADRs
- Base model choice (decided after Phase 5/6 perplexity check)
- Relationship token granularity (`<friend>` vs `<friend_1>` / `<friend_2>`)
- Session gap threshold
- Style classifier negative class

See [DECISIONS.md](DECISIONS.md).
