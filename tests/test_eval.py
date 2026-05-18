"""Tests for the pure-function parts of src/eval.py.

The torch/transformers-dependent functions (perplexity, generation, classifier)
are integration-level and exercised by scripts/07_evaluate.py against a real
trained model. Here we only test the memorization-audit primitives.
"""
from __future__ import annotations

from src.eval import (
    jaccard_tokens,
    lcs_ratio,
    memorization_flag,
    truncate_to_tokens,
)


# ----- Jaccard -----

def test_jaccard_identical():
    assert jaccard_tokens("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert jaccard_tokens("a b c", "d e f") == 0.0


def test_jaccard_partial():
    # tokens {a,b,c} vs {b,c,d} → intersection 2, union 4 → 0.5
    assert jaccard_tokens("a b c", "b c d") == 0.5


def test_jaccard_case_insensitive():
    assert jaccard_tokens("Hello", "hello") == 1.0


def test_jaccard_both_empty():
    assert jaccard_tokens("", "") == 1.0


def test_jaccard_one_empty():
    assert jaccard_tokens("hi", "") == 0.0


# ----- LCS ratio -----

def test_lcs_identical():
    assert lcs_ratio("hello world", "hello world") == 1.0


def test_lcs_disjoint():
    assert lcs_ratio("a b c", "d e f") == 0.0


def test_lcs_word_subsequence():
    # "a b c d" vs "a x c y" → LCS = "a c" → 2/4 = 0.5
    assert lcs_ratio("a b c d", "a x c y") == 0.5


def test_lcs_order_matters():
    # "a b c" vs "c b a" → LCS length 1, max len 3 → 1/3
    assert abs(lcs_ratio("a b c", "c b a") - 1 / 3) < 1e-9


# ----- Memorization flag -----

def test_memorization_flag_below_threshold():
    flag = memorization_flag("the cat sat", "the dog ran")
    # 1 common token ("the") out of 5 unique → low.
    assert not flag["flagged"]


def test_memorization_flag_above_threshold():
    flag = memorization_flag("ariyilla machaane oru manikoor",
                              "ariyilla machaane oru manikoor")
    assert flag["flagged"]
    assert flag["jaccard"] == 1.0
    assert flag["lcs"] == 1.0


def test_memorization_flag_partial_overlap():
    # 80% Jaccard threshold; let's craft a string with 4/5 overlap.
    target = "one two three four five"
    gen = "one two three four six"  # 4 shared, 1 different = jaccard 4/6 ≈ 0.667
    flag = memorization_flag(target, gen, jaccard_threshold=0.8, lcs_threshold=0.8)
    # LCS = 4 ("one two three four"), max len = 5, ratio 0.8 → flagged.
    assert flag["flagged"]


# ----- truncate_to_tokens -----

def test_truncate_to_tokens_shorter_unchanged():
    assert truncate_to_tokens("a b", 5) == "a b"


def test_truncate_to_tokens_clipped():
    assert truncate_to_tokens("a b c d e f g h i", 4) == "a b c d"
