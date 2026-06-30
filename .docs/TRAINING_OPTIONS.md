# Training options — Kaggle vs RunPod

You're about to run Phase 6 (fine-tuning) + Phase 7 (eval). This doc decides **where**.

## Comparison

| | **Kaggle T4** (free) | **RunPod RTX 4090** (paid) |
|---|---|---|
| Cost | **$0** | ~$2.50-3.00 |
| GPU | Tesla T4 (15 GB) | RTX 4090 (24 GB) |
| Precision | fp16 (auto-fallback) | bf16 (native) |
| Wall time, full 3-epoch run | **5-8 hours** | **2-3 hours** |
| Eval (Phase 7) | ~45 min | ~30 min |
| Session limit | 12 hr per session, 30 hr/week quota | none — runs until you stop the pod |
| UX | Notebook (browser cells) | SSH terminal + Jupyter |
| Data upload | Private Kaggle dataset (drag/drop) | scp or runpodctl |
| Adapter download | tar.gz from notebook output | scp |
| Setup hassle | Dataset creation + notebook setup | Pod provisioning + SSH config |
| Iterating on hyperparameters | Quota-limited (each rerun eats your 30 hr) | Pay per minute, no limit |

## Recommendation

**First-time fine-tuner → Kaggle.** Free, slower but fine, the notebook UI gives you a clear running log without needing SSH or terminal skills.

**Tight schedule (you want results today) → RunPod.** ~$3 for a 3× speed-up. Worth it.

**Plan to iterate (e.g. try different LoRA ranks, learning rates) → RunPod.** Kaggle's 30 hr/week quota gets tight quickly when each full run is 6 hours.

For a **portfolio-grade single run**, both produce the same model — pick by cost vs time preference.

## Detailed step-by-step procedures

- **Kaggle:** see [KAGGLE_DEPLOY.md](KAGGLE_DEPLOY.md)
- **RunPod:** see [RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md)

Both guides cover: pre-flight checks, environment setup, data upload, training command, eval, adapter download, common gotchas.

## Pre-flight (do this BEFORE provisioning either GPU)

These steps don't need a GPU and are mandatory before the real training run:

1. **Manual anonymization review.** Open `eval/anon_review_samples.txt` (20 random anonymized targets, gitignored, local only). Scan for any PII the regex missed:
   - Family member names
   - Addresses / locations
   - School / workplace names
   - Account numbers / IDs the regex didn't catch

   If you find any, add the name to `data/anonymized/name_mapping.json` as an alias under the appropriate token (e.g. `"<person_3>": ["fay", "Faiu", "<the missed nickname>"]`), then re-run Phases 2 → 3 → 4.

2. **(Optional)** Drop scraped Manglish text into `data/eval_negatives/*.txt` (one snippet per line). Suggested sources: r/Kerala posts, public Manglish Telegram channels. Without this, the Phase 7 style classifier section just gets skipped (no error). With it, you get a real comparison of "does this sound like me vs. some other Manglish writer."

3. **Confirm files exist locally:**
   - `data/processed/train.jsonl` (~166 MB, 301,612 examples)
   - `data/processed/val.jsonl` (~18 MB, 32,396 examples)
   - `data/anonymized/name_mapping.json` (~1 KB)

## What's already verified and ready

- Pipeline is **reproducible** (Phase 3 produces byte-identical output across runs)
- **115/115 tests pass**, deterministic
- **All 9 scripts** load + `--help` works (parse + import)
- **Smoke training** (1000 examples × 1 epoch, ~28 min) succeeded:
  - Loss dropped 6.47 → 4.23 in original run; 5.42 → 3.27 after the special-token fix
  - Adapter loads correctly and generates recognizable Manglish
  - `python scripts/chat.py` works locally
- **bf16/fp16 auto-detection** in place — T4 (Kaggle) falls back to fp16 automatically; RTX 4090 (RunPod) uses bf16
- **Hyperparameters tuned:**
  - `max_seq_length=256` (real p99 is 170 tokens; this clips only 0.16%)
  - `batch_size=16, grad_accum=1` on 24 GB (RunPod); `batch_size=8, grad_accum=2` on 15 GB (Kaggle)
  - LoRA rank 16, alpha 32, dropout 0.05, all 7 target modules
  - lr 2e-4, cosine schedule, warmup 3%
- **Repetition penalty 1.2** in eval + chat + demo (suppresses "Aaah Aah Aaah" loops we saw in the first smoke run)
- **Repo pushed**: latest commit on `main` at github.com/rihaans/ryGPT

## Cost / time budget — full pipeline projection

**Kaggle path (free):**
- Pre-flight (anonymization review): 10 min (local, no cost)
- Dataset upload to Kaggle: 5 min
- Notebook setup: 5 min
- Training (3 epochs): 6-8 hr (depending on T4 throughput)
- Eval (Phase 7): 45 min
- Download adapter: 2 min
- **Total wall time: ~7-9 hours**, **cost: $0**

**RunPod path (paid):**
- Pre-flight: 10 min (local)
- Pod provisioning: 5 min
- Data upload: 5-10 min
- Training (3 epochs): 2-3 hr
- Eval (Phase 7): 30 min
- Adapter download: 5 min
- **Total wall time: ~3.5-4.5 hr**, **cost: ~$2.50-3.00** (RTX 4090 Secure at $0.69/hr)

## After training, regardless of where

Both paths produce the same artifact: `models/lora_adapter/` (~50 MB) + `eval/*.md`.

You then:

1. Open `eval/memorization.md` — confirm the gate isn't `FAILED`
2. Open `eval/perplexity.md` — check tuned < base across all relationship slices
3. Open `eval/samples.md` — manually annotate which generations sound like you
4. (If you want a demo) `python scripts/08_demo.py` — local Gradio app
5. Or just: `python scripts/chat.py` — interactive CLI chat

Inference on the trained model runs **fine on your 4070 laptop** — no GPU rental needed past Phase 6/7.

## When to revisit decisions

- If the **memorization gate trips** (>5% flagged): train with `--epochs 1` or `--lora-dropout 0.1` and re-eval
- If the **`<group>` perplexity is worse than base**: the group-chat threading issue (ADR concern) is real — may need to revisit Phase 3 context construction
- If **generations are bland / generic**: try `--temperature 0.9` and `--repetition-penalty 1.1` at inference
- If **generations are incoherent**: try `--temperature 0.7` and `--repetition-penalty 1.3`

## TL;DR

```
[you] open eval/anon_review_samples.txt and skim it          ~10 min
[you] pick: Kaggle (free, slower) or RunPod (~$3, faster)
[you] follow KAGGLE_DEPLOY.md  OR  RUNPOD_DEPLOY.md
[you] download models/lora_adapter/ back to laptop
[you] python scripts/chat.py
```
