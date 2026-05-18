# ryGPT — Manglish personal LM

A small fine-tuned language model that writes Manglish (Malayalam in Latin script) in my personal style, conditioned on who I'm talking to.

> Portfolio / personal project. Trained on private WhatsApp chats. No public model release.

## Status
Scaffolding complete. Phase 1 (parsing) not yet started.

## Quick links
- **[PROJECT.md](PROJECT.md)** — phase-by-phase spec (source of truth for ordering and STOP points)
- **[.docs/PRD.md](.docs/PRD.md)** — what we're building and why
- **[.docs/ARCHITECTURE.md](.docs/ARCHITECTURE.md)** — system shape
- **[.docs/DATA_SCHEMA.md](.docs/DATA_SCHEMA.md)** — JSON schemas at every phase boundary
- **[.docs/EVAL_PLAN.md](.docs/EVAL_PLAN.md)** — Phase 7 methodology
- **[.docs/DECISIONS.md](.docs/DECISIONS.md)** — ADR log

## Running the pipeline
Each phase is one script. Run in order:
```bash
python scripts/01_parse_whatsapp.py
python scripts/02_anonymize.py
python scripts/03_build_dataset.py
python scripts/04_data_stats.py          # STOP — review eval/data_stats.md
python scripts/05_train_tokenizer.py     # STOP — review eval/tokenizer_compression.md
python scripts/06_train_model.py         # runs on rented GPU
python scripts/07_evaluate.py
python scripts/08_demo.py                # optional, local-only, gated on memorization audit
```

## Tests
```bash
pytest tests/
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# For Phases 6+, install torch with the right CUDA build on the training box:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# pip install bitsandbytes
```

## Privacy
Raw chat data, anonymized derivatives, and `name_mapping.json` are gitignored. The model is not released publicly without an explicit memorization audit (see [.docs/DECISIONS.md](.docs/DECISIONS.md) ADR-006).
