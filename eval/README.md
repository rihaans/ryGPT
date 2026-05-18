# eval/

Eval outputs land here. Everything except this README and `.gitkeep` is gitignored by default — un-ignore individual files explicitly *after* manual PII review.

Expected outputs (populated by the scripts):
- `data_stats.md` / `data_stats.json` — Phase 4
- `tokenizer_compression.md` — Phase 5
- `perplexity.md` — Phase 7
- `style_classifier.md` — Phase 7
- `samples.md` — Phase 7 (review carefully — contains generated text)
- `memorization.md` — Phase 7 (gate on Phase 8)
- `sample_prompts.json` — fixed prompts used for sample generation, kept for reproducibility

See [.docs/EVAL_PLAN.md](../.docs/EVAL_PLAN.md) for methodology.
