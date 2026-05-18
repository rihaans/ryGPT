"""Tests for src/dataset.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset import (
    build_examples_from_chat,
    build_examples_from_session,
    detect_relationship,
    example_to_chat_messages,
    example_to_prompt_messages,
    make_session_id,
    read_jsonl,
    segment_into_sessions,
    split_train_val,
    write_jsonl,
)


def _msg(ts: str, speaker: str, text: str) -> dict:
    return {"timestamp": ts, "speaker": speaker, "text": text}


# ----- Session segmentation -----

def test_segment_empty():
    assert segment_into_sessions([], gap_hours=2.0) == []


def test_segment_single_message():
    msgs = [_msg("2024-01-01T10:00:00", "<self>", "hi")]
    assert segment_into_sessions(msgs, gap_hours=2.0) == [msgs]


def test_segment_no_gap_one_session():
    msgs = [
        _msg("2024-01-01T10:00:00", "<self>", "a"),
        _msg("2024-01-01T10:30:00", "<person_1>", "b"),
        _msg("2024-01-01T11:00:00", "<self>", "c"),
    ]
    sessions = segment_into_sessions(msgs, gap_hours=2.0)
    assert len(sessions) == 1
    assert sessions[0] == msgs


def test_segment_gap_creates_new_session():
    msgs = [
        _msg("2024-01-01T10:00:00", "<self>", "a"),
        _msg("2024-01-01T10:30:00", "<person_1>", "b"),
        # 3h gap → new session
        _msg("2024-01-01T13:31:00", "<self>", "c"),
    ]
    sessions = segment_into_sessions(msgs, gap_hours=2.0)
    assert len(sessions) == 2
    assert sessions[0][-1]["text"] == "b"
    assert sessions[1][0]["text"] == "c"


def test_segment_gap_exactly_at_threshold_not_split():
    """Threshold uses strict >, so gap exactly at threshold stays in same session."""
    msgs = [
        _msg("2024-01-01T10:00:00", "<self>", "a"),
        _msg("2024-01-01T12:00:00", "<person_1>", "b"),
    ]
    sessions = segment_into_sessions(msgs, gap_hours=2.0)
    assert len(sessions) == 1


def test_segment_unsorted_input_gets_sorted():
    msgs = [
        _msg("2024-01-01T12:00:00", "<self>", "c"),
        _msg("2024-01-01T10:00:00", "<self>", "a"),
        _msg("2024-01-01T11:00:00", "<person_1>", "b"),
    ]
    sessions = segment_into_sessions(msgs, gap_hours=2.0)
    assert [m["text"] for m in sessions[0]] == ["a", "b", "c"]


# ----- Session ID -----

def test_session_id_uses_first_message_date_and_hour():
    session = [_msg("2024-08-14T22:31:07", "<self>", "x")]
    assert make_session_id("fay", session) == "fay_2024-08-14_22"


# ----- Example construction -----

def test_build_examples_from_session_basic():
    session = [
        _msg("2024-01-01T10:00:00", "<person_1>", "where are you"),
        _msg("2024-01-01T10:01:00", "<self>", "at home"),
        _msg("2024-01-01T10:02:00", "<person_1>", "when coming"),
        _msg("2024-01-01T10:03:00", "<self>", "soon"),
    ]
    examples = build_examples_from_session(
        session, session_id="s1", relationship="gf", context_size=8,
    )
    assert len(examples) == 2
    assert examples[0]["target"] == "at home"
    assert examples[0]["relationship_token"] == "<gf>"
    assert examples[0]["context_msg_count"] == 1
    assert examples[1]["target"] == "soon"
    assert examples[1]["context_msg_count"] == 3  # 3 prior messages incl. earlier <self>


def test_build_examples_skips_self_with_no_prior_context():
    """If session starts with <self>, no context → drop that example."""
    session = [
        _msg("2024-01-01T10:00:00", "<self>", "yo"),
        _msg("2024-01-01T10:01:00", "<person_1>", "hi"),
        _msg("2024-01-01T10:02:00", "<self>", "how are you"),
    ]
    examples = build_examples_from_session(session, "s1", "friend", 8)
    assert len(examples) == 1
    assert examples[0]["target"] == "how are you"


def test_build_examples_skips_media_target():
    session = [
        _msg("2024-01-01T10:00:00", "<person_1>", "send pic"),
        _msg("2024-01-01T10:01:00", "<self>", "[media]"),
        _msg("2024-01-01T10:02:00", "<person_1>", "nice"),
        _msg("2024-01-01T10:03:00", "<self>", "ty"),
    ]
    examples = build_examples_from_session(session, "s1", "gf", 8)
    assert [e["target"] for e in examples] == ["ty"]


def test_build_examples_context_window_limited():
    """Context window caps the lookback."""
    session = [_msg(f"2024-01-01T10:{i:02d}:00", "<person_1>", f"m{i}") for i in range(10)]
    session.append(_msg("2024-01-01T10:11:00", "<self>", "target"))
    examples = build_examples_from_session(session, "s1", "gf", context_size=4)
    assert len(examples) == 1
    assert examples[0]["context_msg_count"] == 4
    # Should be the 4 most recent messages
    assert [c["text"] for c in examples[0]["context"]] == ["m6", "m7", "m8", "m9"]


def test_build_examples_context_preserves_speaker():
    session = [
        _msg("2024-01-01T10:00:00", "<person_1>", "a"),
        _msg("2024-01-01T10:01:00", "<self>", "b"),
        _msg("2024-01-01T10:02:00", "<person_1>", "c"),
        _msg("2024-01-01T10:03:00", "<self>", "d"),
    ]
    examples = build_examples_from_session(session, "s1", "gf", 8)
    assert examples[-1]["context"] == [
        {"speaker": "<person_1>", "text": "a"},
        {"speaker": "<self>", "text": "b"},
        {"speaker": "<person_1>", "text": "c"},
    ]


# ----- Relationship detection -----

def test_detect_relationship_gf():
    msgs = [_msg("x", "<self>", "h"), _msg("x", "<person_1>", "h")]
    assert detect_relationship("fay", msgs, gf_chat="fay") == "gf"


def test_detect_relationship_friend_when_one_other_speaker():
    msgs = [_msg("x", "<self>", "h"), _msg("x", "<person_2>", "h")]
    assert detect_relationship("aaron", msgs, gf_chat="fay") == "friend"


def test_detect_relationship_group_when_multiple_others():
    msgs = [
        _msg("x", "<self>", "h"),
        _msg("x", "<person_2>", "h"),
        _msg("x", "<person_5>", "h"),
    ]
    assert detect_relationship("musketeers", msgs, gf_chat="fay") == "group"


# ----- End-to-end per chat -----

def test_build_examples_from_chat_segments_and_emits():
    msgs = [
        _msg("2024-01-01T10:00:00", "<person_1>", "hi"),
        _msg("2024-01-01T10:01:00", "<self>", "hi back"),
        # 3h gap → new session
        _msg("2024-01-01T14:00:00", "<person_1>", "you up"),
        _msg("2024-01-01T14:01:00", "<self>", "yes"),
    ]
    examples = build_examples_from_chat(
        "fay", msgs, relationship="gf", gap_hours=2.0, context_size=8,
    )
    # Two sessions, each contributing one example.
    assert len(examples) == 2
    sids = {e["session_id"] for e in examples}
    assert sids == {"fay_2024-01-01_10", "fay_2024-01-01_14"}


# ----- Train/val split -----

def test_split_is_session_disjoint():
    examples = [
        {"session_id": f"s_{i}", "target": f"m{i}", "relationship": "gf",
         "relationship_token": "<gf>", "context": [{"speaker": "x", "text": "x"}],
         "context_msg_count": 1}
        for i in range(20)
    ]
    train, val = split_train_val(examples, val_fraction=0.1, seed=42)
    train_sids = {e["session_id"] for e in train}
    val_sids = {e["session_id"] for e in val}
    assert train_sids & val_sids == set()  # disjoint
    assert len(val_sids) == 2  # 10% of 20


def test_split_examples_in_same_session_stay_together():
    examples = (
        [{"session_id": "s_a", "target": "x", "relationship": "gf",
          "relationship_token": "<gf>", "context": [], "context_msg_count": 0}
         for _ in range(5)]
        + [{"session_id": "s_b", "target": "y", "relationship": "gf",
            "relationship_token": "<gf>", "context": [], "context_msg_count": 0}
           for _ in range(5)]
    )
    train, val = split_train_val(examples, val_fraction=0.5, seed=0)
    # Each session is entirely in one bucket.
    for bucket in (train, val):
        sids = {e["session_id"] for e in bucket}
        assert len(sids) == 1


def test_split_is_deterministic_with_seed():
    examples = [
        {"session_id": f"s_{i}", "target": "x", "relationship": "gf",
         "relationship_token": "<gf>", "context": [], "context_msg_count": 0}
        for i in range(50)
    ]
    train_a, val_a = split_train_val(examples, val_fraction=0.2, seed=42)
    train_b, val_b = split_train_val(examples, val_fraction=0.2, seed=42)
    assert [e["session_id"] for e in train_a] == [e["session_id"] for e in train_b]
    assert [e["session_id"] for e in val_a] == [e["session_id"] for e in val_b]


def test_split_minimum_one_val_session():
    # 5 sessions × 1% fraction should still allocate at least 1 to val.
    examples = [
        {"session_id": f"s_{i}", "target": "x", "relationship": "gf",
         "relationship_token": "<gf>", "context": [], "context_msg_count": 0}
        for i in range(5)
    ]
    train, val = split_train_val(examples, val_fraction=0.01, seed=0)
    assert len({e["session_id"] for e in val}) >= 1


# ----- JSONL IO -----

def test_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "out.jsonl"
    examples = [
        {"session_id": "s1", "target": "hi", "relationship": "gf",
         "relationship_token": "<gf>", "context": [{"speaker": "<person_1>", "text": "yo"}],
         "context_msg_count": 1},
        {"session_id": "s2", "target": "ok", "relationship": "friend",
         "relationship_token": "<friend>", "context": [{"speaker": "<person_2>", "text": "x"}],
         "context_msg_count": 1},
    ]
    write_jsonl(examples, path)
    loaded = read_jsonl(path)
    assert loaded == examples


def _example(context, target, relationship="gf"):
    return {
        "session_id": "s1",
        "relationship": relationship,
        "relationship_token": f"<{relationship}>",
        "context": context,
        "target": target,
        "context_msg_count": len(context),
    }


# ----- Chat template formatting -----

def test_chat_messages_have_system_relationship_token():
    ex = _example([{"speaker": "<person_3>", "text": "hi"}], "ok")
    msgs = example_to_chat_messages(ex)
    assert msgs[0] == {"role": "system", "content": "<gf>"}


def test_chat_messages_assign_assistant_to_self():
    ex = _example(
        [
            {"speaker": "<person_3>", "text": "where"},
            {"speaker": "<self>", "text": "home"},
        ],
        "soon",
    )
    msgs = example_to_chat_messages(ex)
    # system + user + assistant + assistant(target) = 4
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "assistant"]


def test_chat_messages_merge_consecutive_same_speaker():
    ex = _example(
        [
            {"speaker": "<person_3>", "text": "a"},
            {"speaker": "<person_3>", "text": "b"},
            {"speaker": "<self>", "text": "c"},
        ],
        "d",
    )
    msgs = example_to_chat_messages(ex)
    # Consecutive person_3 turns merged into one user turn.
    user_turns = [m["content"] for m in msgs if m["role"] == "user"]
    assert user_turns == ["<person_3>: a\nb"]


def test_chat_messages_prefix_each_turn_with_speaker():
    ex = _example(
        [{"speaker": "<person_3>", "text": "hi"}],
        "yo",
    )
    msgs = example_to_chat_messages(ex)
    assert msgs[1]["content"] == "<person_3>: hi"
    assert msgs[-1]["content"] == "<self>: yo"


def test_chat_messages_group_chat_distinct_speakers():
    ex = _example(
        [
            {"speaker": "<person_2>", "text": "yo"},
            {"speaker": "<person_5>", "text": "what"},
            {"speaker": "<person_2>", "text": "lol"},
        ],
        "lmao",
        relationship="group",
    )
    msgs = example_to_chat_messages(ex)
    # All three non-self speakers stay separate (different speakers => different turns).
    user_turns = [m["content"] for m in msgs if m["role"] == "user"]
    assert user_turns == ["<person_2>: yo", "<person_5>: what", "<person_2>: lol"]
    assert msgs[0]["content"] == "<group>"


def test_prompt_messages_omits_final_assistant():
    ex = _example(
        [{"speaker": "<person_3>", "text": "hi"}],
        "yo",
    )
    full = example_to_chat_messages(ex)
    prompt = example_to_prompt_messages(ex)
    assert prompt == full[:-1]
    assert prompt[-1]["role"] == "user"


def test_jsonl_handles_unicode(tmp_path: Path):
    path = tmp_path / "out.jsonl"
    examples = [{"session_id": "s1", "target": "ariyilla 😀", "relationship": "gf",
                 "relationship_token": "<gf>", "context": [], "context_msg_count": 0}]
    write_jsonl(examples, path)
    text = path.read_text(encoding="utf-8")
    assert "ariyilla 😀" in text
    assert read_jsonl(path)[0]["target"] == "ariyilla 😀"
