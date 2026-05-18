# Manglish Personal LM — Project Spec

## Overview
Fine-tune a small open-source language model on my WhatsApp chat history to generate Manglish (Malayalam written in Latin script) text in my personal style. Final deliverable: a working model + a clean writeup suitable for a portfolio / resume project.

## Data sources
- 4 WhatsApp chat exports (zip files) in `data/raw/`:
  - 1 chat with girlfriend
  - 2 chats with individual friends
  - 1 group chat
- Each export contains a `_chat.txt` and possibly media files (which we discard).

## Pipeline phases

### Phase 1: Parsing & structuring
- Parse WhatsApp export format. Handle:
  - Both `[DD/MM/YY, HH:MM:SS]` and `DD/MM/YY, HH:MM -` formats (WhatsApp varies by region/version)
  - Multi-line messages (lines that don't start with a timestamp belong to the previous message)
  - `<Media omitted>` lines → keep as `[media]` placeholder
  - System messages ("X added Y", "messages are end-to-end encrypted") → drop
- Output structured JSON per chat: list of `{timestamp, speaker, text}` objects.

### Phase 2: Anonymization
- Replace real names with stable tokens: `<person_1>`, `<person_2>`, etc.
- Maintain a `name_mapping.json` (gitignored, local only).
- Regex-scrub:
  - Phone numbers (Indian format primarily, but general too)
  - Email addresses
  - URLs containing personal identifiers (Google Docs links, etc.)
  - UPI IDs (`*@okhdfc`, `*@paytm`, `*@ybl`, etc.)
  - Numbers that look like account/card numbers (long digit runs)
- Output to `data/anonymized/`.

### Phase 3: Session segmentation & training-example construction
- Split each chat into sessions on >2 hour gaps.
- For each of my outgoing messages, construct a training example:
  - Context: previous N messages (configurable, default 8), each tagged with speaker role
  - Target: my message
  - Metadata: relationship type (`gf` / `friend` / `group`)
- Prepend a relationship token at the start of context: `<gf>`, `<friend_1>`, `<friend_2>`, `<group>`.
- Skip examples where target is `[media]` or empty.
- Output as JSONL in `data/processed/train.jsonl` and `data/processed/val.jsonl` (90/10 split, split by *session* not by message to avoid leakage).

### Phase 4: Data stats & sanity check
- Print summary: total messages, my messages, per-chat breakdown, session counts, avg session length, token length distribution.
- Generate a few sample training examples to manually review for anonymization completeness.

### Phase 5: Custom tokenizer
- Train a BPE tokenizer (using `tokenizers` library) on the Manglish corpus.
- Vocab size: 16k (we're not covering all of English, just Manglish-specific tokens layered on top of base).
- Compare compression ratio vs. the base model's tokenizer on a held-out Manglish sample. Print numbers.
- Decision point: if custom tokenizer compresses Manglish 1.5x+ better, extend base model's vocab with new tokens (vocab expansion + embedding resize). Otherwise, skip and use base tokenizer.

### Phase 6: Fine-tuning
- Base model: start with `Qwen/Qwen2.5-1.5B` or `meta-llama/Llama-3.2-1B` (whichever has better Manglish baseline — test both with a quick perplexity check).
- Method: QLoRA via `peft` + `transformers` + `bitsandbytes`.
- Format: chat template with relationship token as system message, prior turns as user/assistant alternation, target as final assistant turn.
- Hyperparameters (starting point, tune from here):
  - LoRA rank: 16
  - LoRA alpha: 32
  - Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
  - Learning rate: 2e-4
  - Batch size: 4 with gradient accumulation 4 (effective 16)
  - Epochs: 3 (with early stopping on val loss)
  - 4-bit quantization for base model
- Log to Weights & Biases (or tensorboard if W&B not set up).

### Phase 7: Evaluation
- **Perplexity** on held-out val set, broken down per relationship.
- **Style classifier eval**: train a small classifier (logistic regression on TF-IDF, or a tiny BERT) to distinguish my messages from scraped Manglish from Reddit r/Kerala. Then test: does the classifier think the model's generations are "me"?
- **Generation samples**: For each relationship token, generate 10 sample replies given fixed contexts. Save to `eval/samples.md` for manual review.
- **Memorization check**: For each training example, check if the model's generation has >80% token overlap with the training target. Flag examples where this happens — that's memorization, not generalization.

### Phase 8: Demo (optional)
- Simple Gradio app: pick a relationship, type a message as if you're the other person, get a reply in my style.
- Local-only by default. Don't deploy publicly unless memorization check is clean.

## Constraints & non-goals
- **Privacy**: anonymized data only in version control. Raw data and `name_mapping.json` are gitignored. Never commit chat content.
- **No public model release** unless explicit decision later, after memorization audit.
- Not building: multi-modal, voice, RLHF, agent functionality. Keep scope tight.

## Tech stack
- Python 3.11+
- `transformers`, `peft`, `bitsandbytes`, `accelerate`, `trl`
- `tokenizers` for custom BPE
- `datasets` for data loading
- `wandb` for logging (optional)
- `gradio` for the optional demo

## Hardware
- Training on a rented H100 or A100 (RunPod / Lambda / Vast.ai). Expect ~2-4 hours per training run for a 1.5B model + LoRA.
- Local dev on whatever I have (CPU is fine for data pipeline).

## File structure (target)
```
manglish-lm/
├── PROJECT.md                  (this file)
├── README.md                   (writeup, for portfolio)
├── .gitignore                  (excludes data/raw, data/anonymized, name_mapping.json, models/)
├── requirements.txt
├── scripts/
│   ├── 01_parse_whatsapp.py
│   ├── 02_anonymize.py
│   ├── 03_build_dataset.py
│   ├── 04_data_stats.py
│   ├── 05_train_tokenizer.py
│   ├── 06_train_model.py
│   ├── 07_evaluate.py
│   └── 08_demo.py
├── src/
│   ├── parsing.py
│   ├── anonymize.py
│   ├── dataset.py
│   └── eval.py
├── tests/
│   └── (unit tests for parsing & anonymization — these MUST exist, parsing edge cases are sneaky)
├── data/
│   ├── raw/            (gitignored)
│   ├── anonymized/     (gitignored)
│   └── processed/      (gitignored)
├── models/             (gitignored)
└── eval/
    └── samples.md
```

## Order of operations for Claude Code
1. Set up project structure, requirements.txt, .gitignore
2. Build & test the WhatsApp parser (Phase 1) — write tests with fake export data first
3. Build anonymization (Phase 2) — also with tests
4. Build dataset construction (Phase 3)
5. Run data stats (Phase 4) — STOP here, let me review outputs before training
6. Train tokenizer (Phase 5) — STOP, review compression numbers
7. Train model (Phase 6) — I'll run this on rented GPU
8. Eval (Phase 7)
9. Optional demo (Phase 8)

At each STOP point, summarize what was built and ask before proceeding.
