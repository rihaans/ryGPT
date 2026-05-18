"""Tests for src/anonymize.py. All fixtures are synthetic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.anonymize import (
    _build_replacement_regex,
    anonymize_messages,
    build_mapping_from_speakers,
    collect_speakers,
    load_mapping,
    replace_names,
    save_mapping,
    scrub_text,
)


# ----- PII scrub -----

def test_email_scrubbed():
    assert scrub_text("contact me at foo@bar.com please") == "contact me at [email] please"


def test_email_with_plus_and_dots():
    assert scrub_text("foo.bar+baz@example.co.uk") == "[email]"


def test_upi_paytm():
    assert scrub_text("send to abc@paytm now") == "send to [upi] now"


def test_upi_various_providers():
    for handle in ("user.name@okhdfc", "shop-1@ybl", "x@oksbi", "biz@axisbank"):
        scrubbed = scrub_text(f"pay {handle}")
        assert scrubbed.endswith("[upi]"), f"Failed: {handle} -> {scrubbed}"


def test_indian_mobile_scrubbed():
    assert scrub_text("call 9876543210 ok") == "call [phone] ok"


def test_indian_mobile_only_6_to_9_leading():
    # Numbers starting 0-5 are not Indian mobile — leave them alone.
    assert scrub_text("code 5234567890") == "code 5234567890"


def test_intl_phone_with_spaces():
    assert scrub_text("call +91 98765 43210") == "call [phone]"


def test_intl_phone_with_hyphens():
    assert scrub_text("ring +1-555-123-4567") == "ring [phone]"


def test_intl_phone_with_narrow_nbsp():
    # WhatsApp's contact phone-number display uses narrow NBSP between digit groups.
    chat = "saw +91 87788 90419 joined"
    assert scrub_text(chat) == "saw [phone] joined"


def test_long_digit_run():
    assert scrub_text("card 1234567890123456 expires") == "card [number] expires"


def test_short_digit_runs_not_scrubbed():
    # 11 digits — too short to be a card number, too long to be a phone (and doesn't start 6-9).
    assert scrub_text("ref 12345678901") == "ref 12345678901"


def test_google_docs_link_scrubbed():
    assert scrub_text("see https://docs.google.com/document/d/ABCxyz/edit") == "see [link]"


def test_google_drive_link_scrubbed():
    assert scrub_text("file https://drive.google.com/file/d/XYZ/view") == "file [link]"


def test_generic_url_kept_by_default():
    text = "see https://example.com/page"
    assert scrub_text(text) == text


def test_generic_url_scrubbed_when_disabled():
    assert scrub_text("see https://example.com/page", keep_urls=False) == "see [link]"


# ----- Name mapping construction -----

def test_self_mapping_always_present():
    mapping = build_mapping_from_speakers([], self_name="rihaan")
    assert mapping["<self>"] == ["rihaan"]


def test_speakers_assigned_alphabetically():
    mapping = build_mapping_from_speakers(
        ["fay", "Aaron", "Johan Deepak"], self_name="rihaan"
    )
    # Case-insensitive sort: Aaron, fay, Johan Deepak
    assert mapping["<person_1>"] == ["Aaron"]
    assert mapping["<person_2>"] == ["fay"]
    assert mapping["<person_3>"] == ["Johan Deepak"]


def test_self_name_excluded_from_person_tokens():
    mapping = build_mapping_from_speakers(["rihaan", "Aaron"], self_name="rihaan")
    assert mapping["<self>"] == ["rihaan"]
    assert mapping["<person_1>"] == ["Aaron"]
    assert len([k for k in mapping if k.startswith("<person_")]) == 1


def test_existing_mapping_preserved():
    existing = {"<self>": ["rihaan"], "<person_1>": ["Aaron"]}
    mapping = build_mapping_from_speakers(
        ["Aaron", "fay"], self_name="rihaan", existing=existing
    )
    # Aaron stays at person_1, fay is added as person_2.
    assert mapping["<person_1>"] == ["Aaron"]
    assert mapping["<person_2>"] == ["fay"]


def test_existing_aliases_match_case_insensitively():
    existing = {"<person_1>": ["Aaron"]}
    mapping = build_mapping_from_speakers(
        ["AARON", "fay"], self_name="rihaan", existing=existing
    )
    # AARON shouldn't get a new token.
    assert mapping["<person_1>"] == ["Aaron"]
    assert mapping["<person_2>"] == ["fay"]


def test_mapping_indices_continue_past_existing_max():
    existing = {"<self>": ["rihaan"], "<person_5>": ["Aaron"]}
    mapping = build_mapping_from_speakers(
        ["Aaron", "fay"], self_name="rihaan", existing=existing
    )
    # New tokens start at 6, not 1.
    assert mapping["<person_6>"] == ["fay"]


# ----- Name replacement in text -----

def _regex_for(mapping: dict[str, list[str]]):
    return _build_replacement_regex(mapping)


def test_replace_single_name():
    mapping = {"<person_1>": ["Fay"]}
    regex, lookup = _regex_for(mapping)
    assert replace_names("hey Fay", regex, lookup) == "hey <person_1>"


def test_replace_case_insensitive():
    mapping = {"<person_1>": ["Fay"]}
    regex, lookup = _regex_for(mapping)
    assert replace_names("hey FAY and fay and Fay", regex, lookup) == "hey <person_1> and <person_1> and <person_1>"


def test_replace_word_boundary_substring_safe():
    """Faye should NOT be replaced when matching Fay."""
    mapping = {"<person_1>": ["Fay"]}
    regex, lookup = _regex_for(mapping)
    assert replace_names("Faye walked by", regex, lookup) == "Faye walked by"


def test_replace_multi_word_name():
    mapping = {"<person_1>": ["Johan Deepak"]}
    regex, lookup = _regex_for(mapping)
    assert replace_names("tell Johan Deepak", regex, lookup) == "tell <person_1>"


def test_longer_alias_wins_over_shorter():
    mapping = {"<person_1>": ["Johan Deepak"], "<person_2>": ["Johan"]}
    regex, lookup = _regex_for(mapping)
    # "Johan Deepak" should match before standalone "Johan".
    assert replace_names("Johan Deepak said", regex, lookup) == "<person_1> said"


def test_aliases_for_same_token_both_replaced():
    """Nicknames mapped to the same token both resolve to that token."""
    mapping = {"<person_1>": ["Fay", "Faiu"]}
    regex, lookup = _regex_for(mapping)
    assert replace_names("Fay said, Faiu agreed", regex, lookup) == "<person_1> said, <person_1> agreed"


def test_possessive_apostrophe_preserved():
    mapping = {"<person_1>": ["Fay"]}
    regex, lookup = _regex_for(mapping)
    assert replace_names("Fay's bag", regex, lookup) == "<person_1>'s bag"


def test_empty_mapping_no_change():
    regex, lookup = _regex_for({})
    assert replace_names("Fay and Aaron", regex, lookup) == "Fay and Aaron"


# ----- Full message anonymization -----

def test_anonymize_speaker_field():
    mapping = {"<self>": ["rihaan"], "<person_1>": ["fay"]}
    msgs = [
        {"timestamp": "2024-01-01T00:00:00", "speaker": "rihaan", "text": "hi"},
        {"timestamp": "2024-01-01T00:00:10", "speaker": "fay", "text": "hello"},
    ]
    out = anonymize_messages(msgs, mapping)
    assert out[0]["speaker"] == "<self>"
    assert out[1]["speaker"] == "<person_1>"


def test_anonymize_text_names_and_pii_combined():
    mapping = {"<self>": ["rihaan"], "<person_1>": ["Fay"]}
    msgs = [
        {
            "timestamp": "2024-01-01T00:00:00",
            "speaker": "rihaan",
            "text": "hey Fay, call me at 9876543210 or email me at foo@bar.com",
        }
    ]
    out = anonymize_messages(msgs, mapping)
    assert out[0]["text"] == "hey <person_1>, call me at [phone] or email me at [email]"


def test_unknown_speaker_passes_through():
    mapping = {"<self>": ["rihaan"]}
    msgs = [{"timestamp": "2024-01-01T00:00:00", "speaker": "Stranger", "text": "hi"}]
    out = anonymize_messages(msgs, mapping)
    assert out[0]["speaker"] == "Stranger"


def test_media_placeholder_passes_through():
    mapping = {"<self>": ["rihaan"]}
    msgs = [{"timestamp": "2024-01-01T00:00:00", "speaker": "rihaan", "text": "[media]"}]
    out = anonymize_messages(msgs, mapping)
    assert out[0]["text"] == "[media]"


# ----- Load / save round-trip -----

def test_mapping_save_load_roundtrip(tmp_path: Path):
    path = tmp_path / "name_mapping.json"
    mapping = {"<self>": ["rihaan"], "<person_1>": ["Fay", "Faiu"]}
    save_mapping(mapping, path)
    loaded = load_mapping(path)
    assert loaded == mapping


def test_load_missing_returns_empty(tmp_path: Path):
    assert load_mapping(tmp_path / "nope.json") == {}


# ----- Speaker collection -----

def test_collect_speakers(tmp_path: Path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps([
        {"timestamp": "x", "speaker": "Alice", "text": "hi"},
        {"timestamp": "x", "speaker": "Bob", "text": "hi"},
    ]))
    f2.write_text(json.dumps([
        {"timestamp": "x", "speaker": "alice", "text": "hi"},
        {"timestamp": "x", "speaker": "Carol", "text": "hi"},
    ]))
    speakers = collect_speakers([f1, f2])
    # Sorted case-insensitive, unique by exact string (so "Alice" != "alice").
    assert speakers == ["Alice", "alice", "Bob", "Carol"]


# ----- Real-world phone-number-as-speaker edge case -----

def test_phone_number_speaker_handled_as_alias(tmp_path: Path):
    """A real-world group chat had a contact whose display name was '+91 87788 90419'
    (with narrow NBSPs). It should map to a person token like any other name."""
    speaker = "+91 87788 90419"
    mapping = build_mapping_from_speakers([speaker, "Aaron"], self_name="rihaan")
    # Phone-number speaker should get a person token.
    person_tokens = {tok: aliases for tok, aliases in mapping.items() if tok.startswith("<person_")}
    assert any(speaker in aliases for aliases in person_tokens.values())
