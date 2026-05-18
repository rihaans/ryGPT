"""Evaluation: perplexity, style classifier, generation samples, memorization audit.

Phase 7. See `.docs/EVAL_PLAN.md` for full methodology.

This module deliberately has *no* hard dependency on torch/transformers at
import time — heavy deps are imported lazily inside the functions that need
them, so unit tests and pipeline-only environments don't need a GPU stack.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable


# ---------- Memorization audit ----------

def _tokenize_words(text: str) -> list[str]:
    return text.lower().split()


def jaccard_tokens(a: str, b: str) -> float:
    """Jaccard similarity over whitespace-split token *sets*."""
    sa = set(_tokenize_words(a))
    sb = set(_tokenize_words(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lcs_ratio(a: str, b: str) -> float:
    """Longest common subsequence length / max(len(a_tokens), len(b_tokens))."""
    ta, tb = _tokenize_words(a), _tokenize_words(b)
    if not ta or not tb:
        return 1.0 if (not ta and not tb) else 0.0
    n, m = len(ta), len(tb)
    # Rolling 1D DP for memory efficiency.
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if ta[i - 1] == tb[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m] / max(n, m)


def memorization_flag(
    target: str,
    generation: str,
    jaccard_threshold: float = 0.8,
    lcs_threshold: float = 0.8,
) -> dict:
    j = jaccard_tokens(target, generation)
    l = lcs_ratio(target, generation)
    return {
        "jaccard": j,
        "lcs": l,
        "flagged": (j >= jaccard_threshold) or (l >= lcs_threshold),
    }


# ---------- Perplexity ----------

def compute_perplexity(
    model,
    tokenizer,
    examples: list[dict],
    max_seq_length: int = 1024,
    batch_size: int = 4,
):
    """Cross-entropy averaged over TARGET tokens only, per relationship."""
    import torch
    from src.dataset import example_to_chat_messages

    model.eval()
    device = next(model.parameters()).device
    sum_nll = Counter()
    n_tokens = Counter()
    sum_nll_all = 0.0
    n_tokens_all = 0

    with torch.no_grad():
        for ex in examples:
            messages = example_to_chat_messages(ex)
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            full_ids = tokenizer(
                full_text,
                truncation=True, max_length=max_seq_length,
                add_special_tokens=False, return_tensors="pt",
            )["input_ids"].to(device)

            prefix_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
            )
            prefix_len = len(tokenizer(
                prefix_text, truncation=True, max_length=max_seq_length,
                add_special_tokens=False,
            )["input_ids"])

            if full_ids.shape[1] <= prefix_len:
                continue  # target was truncated away

            labels = full_ids.clone()
            labels[:, :prefix_len] = -100

            out = model(input_ids=full_ids, labels=labels)
            # transformers loss is already mean across non-ignored tokens.
            n_t = (labels != -100).sum().item()
            nll = out.loss.item() * n_t

            rel = ex["relationship"]
            sum_nll[rel] += nll
            n_tokens[rel] += n_t
            sum_nll_all += nll
            n_tokens_all += n_t

    result = {
        "overall": {
            "nll_per_token": sum_nll_all / max(n_tokens_all, 1),
            "perplexity": math.exp(sum_nll_all / max(n_tokens_all, 1)) if n_tokens_all else float("inf"),
            "n_tokens": n_tokens_all,
        }
    }
    for rel in sum_nll:
        nll_per = sum_nll[rel] / max(n_tokens[rel], 1)
        result[rel] = {
            "nll_per_token": nll_per,
            "perplexity": math.exp(nll_per),
            "n_tokens": n_tokens[rel],
        }
    return result


# ---------- Generation ----------

def generate_response(
    model,
    tokenizer,
    example: dict,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int | None = None,
) -> str:
    """Generate a single response from a prompt-shaped example."""
    import torch
    from src.dataset import example_to_prompt_messages

    if seed is not None:
        torch.manual_seed(seed)

    messages = example_to_prompt_messages(example)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------- Style classifier ----------

def train_style_classifier(positive_texts: Iterable[str], negative_texts: Iterable[str]):
    """TF-IDF + LogisticRegression. Returns (classifier, vectorizer, holdout_accuracy)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    pos = list(positive_texts)
    neg = list(negative_texts)
    X = pos + neg
    y = [1] * len(pos) + [0] * len(neg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_train)
    holdout_acc = clf.score(Xte, y_test)
    return clf, vec, holdout_acc


def classify_as_me(classifier, vectorizer, texts: list[str]) -> float:
    """Returns the fraction of texts the classifier labels as positive (=me)."""
    if not texts:
        return 0.0
    X = vectorizer.transform(texts)
    preds = classifier.predict(X)
    return float(sum(preds)) / len(preds)


def truncate_to_tokens(text: str, n: int) -> str:
    """Length-controlled variant: keep only the first n whitespace-tokens."""
    return " ".join(text.split()[:n])
