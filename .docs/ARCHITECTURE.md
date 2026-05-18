# Architecture — ryGPT

## System shape
Offline batch pipeline. No long-running services. Each phase is an idempotent script that reads from one directory and writes to the next. State lives entirely on disk between phases — no DB, no message queue, no daemon. Restartable from any phase boundary.

```
data/raw/*.zip
       │
       ▼  scripts/01_parse_whatsapp.py
data/parsed/*.json                       ← {timestamp, speaker, text} per chat
       │
       ▼  scripts/02_anonymize.py
data/anonymized/*.json   +  name_mapping.json (LOCAL ONLY, gitignored)
       │
       ▼  scripts/03_build_dataset.py
data/processed/{train,val}.jsonl         ← {context[], target, relationship}
       │
       ├──▶ scripts/04_data_stats.py  →  eval/data_stats.md             [STOP]
       │
       ▼  scripts/05_train_tokenizer.py
models/tokenizer/                        + eval/tokenizer_compression.md [STOP]
       │
       ▼  scripts/06_train_model.py (rented GPU)
models/lora_adapter/                     + W&B run logs
       │
       ▼  scripts/07_evaluate.py
eval/{perplexity.md, style_classifier.md, samples.md, memorization.md}
       │
       ▼  scripts/08_demo.py (optional, local-only)
gradio app on localhost
```

## Component responsibilities

| Module                 | Responsibility                                                                 | Pure / Stateful |
|------------------------|--------------------------------------------------------------------------------|-----------------|
| `src/parsing.py`       | WhatsApp `_chat.txt` → list of message records. Handles both timestamp formats, multi-line messages, system messages, media placeholders. | Pure |
| `src/anonymize.py`     | Name replacement + regex PII scrub (phones, emails, UPI, URLs, long digit runs). Emits stable name → token mapping. | Pure given a seeded mapping |
| `src/dataset.py`       | Session segmentation (gap-based), context window construction, relationship tagging, train/val split by session. | Pure given seed |
| `src/eval.py`          | Perplexity, style classifier training + eval, generation sampling, memorization scoring. | Reads model + data |
| `scripts/0X_*.py`      | Thin CLI wrappers. Argparse, IO, logging. No business logic. | Thin |

## Why scripts + library split
Each `scripts/0X_*.py` is a CLI entry point for one phase. The logic lives in `src/` and is import-tested. This means:
- Tests can hit the library directly with fake fixtures, no IO mocking.
- Re-running a phase is one command.
- A reader of the repo can read scripts top-to-bottom to understand the pipeline at a glance.

## State boundaries

**Local-only, never committed:**
- `data/raw/` — original WhatsApp zips
- `data/parsed/`, `data/anonymized/`, `data/processed/` — derivatives still contain personal info
- `name_mapping.json` — reverses anonymization, lives next to anonymized data
- `models/` — base model weights, LoRA adapter, tokenizer
- W&B local cache

**Committable:**
- All source under `src/`, `scripts/`, `tests/`
- `.docs/` — these docs
- `eval/*.md` — summary numbers ONLY (no sample text from chats; eval/samples.md may contain generations but must be reviewed before commit)
- `PROJECT.md`, `README.md`, `requirements.txt`, `.gitignore`

The `.gitignore` enforces this; CI (if added) should also fail on suspicious commits.

## Determinism
- All randomness behind a single `--seed` arg, plumbed to: train/val split, classifier training, generation sampling.
- Same seed + same data → same output at every phase. No timestamps in filenames.

## Failure modes the architecture defends against
1. **Re-running a phase loses work** — every phase writes to its own directory, never overwrites the input. Safe to re-run.
2. **Anonymization regression silently leaks PII** — Phase 4 stats includes "samples for manual anonymization review." Plus tests in `tests/test_anonymize.py` exercise known PII patterns.
3. **Train/val leakage** — split is by *session*, not message (per PROJECT.md Phase 3). Enforced in `src/dataset.py`, asserted in tests.
4. **Memorization in deployed model** — Phase 7 hard gate. No demo if >5% of training examples are reproduced verbatim.

## What's intentionally NOT in the architecture
- No database. JSON/JSONL on disk is sufficient for this volume.
- No async / workers. Sequential, deterministic, restartable is the win.
- No Docker for the data pipeline. Conda/venv is enough. (Training environment on rented GPU is a separate concern, scripted ad-hoc.)
- No "framework." `transformers` + `peft` is the framework.
