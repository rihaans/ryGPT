"""Evaluation: perplexity, style classifier, generation samples, memorization audit.

Phase 7. See `.docs/EVAL_PLAN.md` for full methodology.

Functions to implement (one per metric):
- perplexity_per_relationship(model, val_examples) -> dict
- train_style_classifier(positive, negatives_dict) -> classifier  # multiple negative classes per ADR-003
- classify_generations(classifier, generations) -> dict
- generate_samples(model, fixed_contexts, sampling_params) -> list
- memorization_audit(model, train_examples, threshold=0.8) -> list[FlaggedExample]
"""
