"""Tests for src/parsing.py. All fixtures are synthetic — no real chat content."""
from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path

import pytest

from src.parsing import (
    chat_name_from_zip_filename,
    parse_chat_text,
    parse_chat_zip,
)


# ----- Timestamp formats -----

def test_parses_ios_format_with_seconds():
    chat = "[14/08/24, 22:31:07] Alice: hello\n"
    msgs = parse_chat_text(chat)
    assert msgs == [
        {"timestamp": "2024-08-14T22:31:07", "speaker": "Alice", "text": "hello"}
    ]


def test_parses_android_format_without_seconds():
    chat = "14/08/24, 22:31 - Alice: hello\n"
    msgs = parse_chat_text(chat)
    assert msgs == [
        {"timestamp": "2024-08-14T22:31:00", "speaker": "Alice", "text": "hello"}
    ]


def test_parses_ios_with_am_pm():
    chat = "[14/08/24, 10:31:07 PM] Alice: hello\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["timestamp"] == "2024-08-14T22:31:07"


def test_parses_android_with_am_pm():
    chat = "14/08/24, 10:31 AM - Alice: hello\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["timestamp"] == "2024-08-14T10:31:00"


def test_parses_four_digit_year():
    chat = "[14/08/2024, 22:31:07] Alice: hello\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["timestamp"] == "2024-08-14T22:31:07"


def test_narrow_no_break_space_before_am_pm():
    # WhatsApp iOS occasionally uses U+202F (narrow NBSP) before AM/PM.
    chat = "[14/08/24, 10:31:07 PM] Alice: hello\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["timestamp"] == "2024-08-14T22:31:07"


# ----- Multi-line messages -----

def test_multiline_message_joined_with_newlines():
    chat = textwrap.dedent("""\
    [14/08/24, 22:31:07] Alice: line one
    line two
    line three
    [14/08/24, 22:31:22] Bob: reply
    """)
    msgs = parse_chat_text(chat)
    assert len(msgs) == 2
    assert msgs[0]["text"] == "line one\nline two\nline three"
    assert msgs[1]["text"] == "reply"


def test_continuation_after_system_message_is_discarded():
    """A continuation line following a system message should NOT attach to a previous message."""
    chat = textwrap.dedent("""\
    [14/08/24, 22:30:00] Alice: hi
    [14/08/24, 22:30:30] Bob created group
    stray continuation line
    [14/08/24, 22:31:00] Alice: ok
    """)
    msgs = parse_chat_text(chat)
    assert [m["text"] for m in msgs] == ["hi", "ok"]


# ----- System messages -----

def test_e2e_encryption_notice_dropped():
    chat = "[14/08/24, 22:30:00] Messages and calls are end-to-end encrypted.\n"
    assert parse_chat_text(chat) == []


def test_user_added_event_dropped():
    chat = "[14/08/24, 22:30:00] Alice added Bob\n"
    assert parse_chat_text(chat) == []


def test_group_lifecycle_events_with_group_name_speaker_dropped():
    """Group exports prefix lifecycle events with the GROUP NAME as 'speaker' — must still drop."""
    chat = textwrap.dedent("""\
    [14/08/24, 22:30:00] Musketeers: Messages and calls are end-to-end encrypted. Tap to learn more.
    [14/08/24, 22:30:01] Musketeers: You created group "Musketeers"
    [14/08/24, 22:30:02] Musketeers: You changed this group's icon
    [14/08/24, 22:31:00] Alice: hi
    """)
    msgs = parse_chat_text(chat)
    assert [m["text"] for m in msgs] == ["hi"]


def test_subject_change_dropped():
    chat = "[14/08/24, 22:30:00] Alice: You changed the subject from \"X\" to \"Y\"\n"
    assert parse_chat_text(chat) == []


def test_security_code_change_dropped():
    chat = "[14/08/24, 22:30:00] Alice: Your security code with Bob has changed.\n"
    assert parse_chat_text(chat) == []


def test_joined_via_invite_link_dropped():
    chat = "[14/08/24, 22:30:00] Alice: Bob joined using this group's invite link\n"
    assert parse_chat_text(chat) == []


def test_only_system_messages_yields_empty_list():
    chat = textwrap.dedent("""\
    [14/08/24, 22:30:00] Messages and calls are end-to-end encrypted.
    [14/08/24, 22:30:01] Alice added Bob
    [14/08/24, 22:30:02] Bob changed the subject to "test"
    """)
    assert parse_chat_text(chat) == []


# ----- Media handling -----

def test_media_omitted_becomes_placeholder():
    chat = "[14/08/24, 22:31:07] Alice: <Media omitted>\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["text"] == "[media]"


def test_image_omitted_android_becomes_placeholder():
    chat = "14/08/24, 22:31 - Alice: image omitted\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["text"] == "[media]"


def test_file_attached_becomes_media():
    chat = "[14/08/24, 22:31:07] Alice: IMG-20240814-WA0001.jpg (file attached)\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["text"] == "[media]"


def test_sticker_audio_video_document_gif_all_become_media():
    for kind in ("sticker omitted", "audio omitted", "video omitted",
                 "document omitted", "GIF omitted"):
        chat = f"14/08/24, 22:31 - Alice: {kind}\n"
        msgs = parse_chat_text(chat)
        assert msgs[0]["text"] == "[media]", f"Failed for: {kind}"


# ----- Edited / deleted -----

def test_edited_marker_stripped_content_kept():
    chat = "[14/08/24, 22:31:07] Alice: hello there <This message was edited>\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["text"] == "hello there"


def test_deleted_message_dropped():
    chat = textwrap.dedent("""\
    [14/08/24, 22:31:07] Alice: This message was deleted
    [14/08/24, 22:31:08] Bob: ok
    """)
    msgs = parse_chat_text(chat)
    assert [m["speaker"] for m in msgs] == ["Bob"]


def test_you_deleted_message_dropped():
    chat = "[14/08/24, 22:31:07] Alice: You deleted this message\n"
    assert parse_chat_text(chat) == []


def test_message_only_edited_marker_yields_empty_and_drops():
    chat = "[14/08/24, 22:31:07] Alice: <This message was edited>\n"
    assert parse_chat_text(chat) == []


# ----- Unicode / edge cases -----

def test_emoji_preserved():
    chat = "[14/08/24, 22:31:07] Alice: hello 😀 nice\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["text"] == "hello 😀 nice"


def test_directional_marks_stripped():
    # iOS exports sometimes prefix lines with LRM (U+200E).
    chat = "‎[14/08/24, 22:31:07] Alice: ‎hello‎\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["text"] == "hello"


def test_speaker_name_with_hyphen_android():
    """`(.*)` after Android `' - '` is greedy, so `Speaker - Name: text` partitions correctly."""
    chat = "14/08/24, 22:31 - Alice - Old Phone: hello\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["speaker"] == "Alice - Old Phone"
    assert msgs[0]["text"] == "hello"


def test_colon_in_message_preserved():
    chat = "[14/08/24, 22:31:07] Alice: meet at 10:30 sharp\n"
    msgs = parse_chat_text(chat)
    assert msgs[0]["text"] == "meet at 10:30 sharp"


def test_empty_input_yields_empty_list():
    assert parse_chat_text("") == []


def test_whitespace_only_input_yields_empty_list():
    assert parse_chat_text("\n\n\n") == []


def test_mixed_real_and_system_messages():
    chat = textwrap.dedent("""\
    [14/08/24, 22:30:00] Messages and calls are end-to-end encrypted.
    [14/08/24, 22:30:30] Alice: hi
    [14/08/24, 22:30:35] Bob: hey
    [14/08/24, 22:31:00] Alice added Carol
    [14/08/24, 22:31:30] Carol: hello everyone
    """)
    msgs = parse_chat_text(chat)
    assert [m["speaker"] for m in msgs] == ["Alice", "Bob", "Carol"]
    assert [m["text"] for m in msgs] == ["hi", "hey", "hello everyone"]


# ----- ZIP-level integration -----

def test_parse_chat_zip_finds_underscore_chat_txt(tmp_path: Path):
    zip_path = tmp_path / "WhatsApp Chat - Test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("_chat.txt", "[14/08/24, 22:31:07] Alice: hello from zip\n")
    msgs = parse_chat_zip(zip_path)
    assert len(msgs) == 1
    assert msgs[0]["text"] == "hello from zip"


def test_parse_chat_zip_falls_back_to_any_txt(tmp_path: Path):
    zip_path = tmp_path / "WhatsApp Chat - Test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("WhatsApp Chat with Alice.txt", "14/08/24, 22:31 - Alice: hi\n")
    msgs = parse_chat_zip(zip_path)
    assert msgs[0]["text"] == "hi"


def test_parse_chat_zip_raises_if_no_txt(tmp_path: Path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("image.jpg", b"\xff\xd8\xff")
    import pytest
    with pytest.raises(FileNotFoundError):
        parse_chat_zip(zip_path)


# ----- Filename → slug -----

def test_chat_name_from_filename_strips_prefix_and_lowercases():
    assert chat_name_from_zip_filename(Path("WhatsApp Chat - Fay.zip")) == "fay"
    assert chat_name_from_zip_filename(Path("WhatsApp Chat - Johan Deepak.zip")) == "johan_deepak"
    assert chat_name_from_zip_filename(Path("WhatsApp Chat - Musketeers.zip")) == "musketeers"


def test_chat_name_from_filename_without_prefix():
    assert chat_name_from_zip_filename(Path("Some Other Name.zip")) == "some_other_name"
