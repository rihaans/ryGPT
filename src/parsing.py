"""WhatsApp chat export parser.

Phase 1 of the pipeline. Reads `_chat.txt` files (extracted from the WhatsApp
zip exports) and produces a list of {timestamp, speaker, text} records.

Handles:
- Both `[DD/MM/YY, HH:MM:SS]` (iOS-style) and `DD/MM/YY, HH:MM -` (Android-style) timestamps.
- 12-hour (AM/PM) and 24-hour times; 2-digit and 4-digit years.
- Multi-line messages (continuation lines lack a leading timestamp).
- Media omitted / file attached → `[media]` placeholder.
- Edited-message markers stripped, edited content kept.
- Deleted-message placeholders dropped.
- System messages dropped (lines after timestamp without `Name: text` shape).
- Unicode directional marks (LRM/RLM/PDF/etc.) stripped.

See `.docs/DATA_SCHEMA.md` for the output schema.
"""
from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

# WhatsApp iOS exports sprinkle bidi/format control marks throughout the text.
# Strip these before regex matching so a stray LRM doesn't kill a header match.
_INVISIBLES = "‎‏‪‫‬‭‮﻿"
_DIRECTIONAL_MARKS = re.compile(f"[{_INVISIBLES}]")

# Some exports use NARROW NO-BREAK SPACE (U+202F) or NBSP (U+00A0) between time and AM/PM.
_TIMESTAMP_SPACE_NORMALIZE = {" ": " ", "\xa0": " "}

# Timestamp shape: interior of the iOS bracket, or before the Android " - ".
# Day/month: 1-2 digits. Year: 2 or 4 digits. Time: H:MM or H:MM:SS, optional AM/PM.
_TIMESTAMP_RE = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?"

_IOS_HEADER = re.compile(rf"^\[({_TIMESTAMP_RE})\]\s*(.*)$")
_ANDROID_HEADER = re.compile(rf"^({_TIMESTAMP_RE})\s-\s(.*)$")

# Tried in order; first one that parses wins.
_TIMESTAMP_FORMATS = (
    "%d/%m/%y, %H:%M:%S",
    "%d/%m/%Y, %H:%M:%S",
    "%d/%m/%y, %H:%M",
    "%d/%m/%Y, %H:%M",
    "%d/%m/%y, %I:%M:%S %p",
    "%d/%m/%Y, %I:%M:%S %p",
    "%d/%m/%y, %I:%M %p",
    "%d/%m/%Y, %I:%M %p",
)

# Whole-message text patterns that map to the `[media]` placeholder.
_MEDIA_PATTERNS = re.compile(
    r"^("
    r"<Media omitted>"
    r"|image omitted"
    r"|video omitted"
    r"|audio omitted"
    r"|sticker omitted"
    r"|document omitted"
    r"|GIF omitted"
    r"|.+\(file attached\)"
    r")$",
    re.IGNORECASE,
)

_DELETION_TEXTS = (
    "This message was deleted",
    "You deleted this message",
    "null",
)

# Whole-message text patterns for system events that arrive with a Speaker: shape
# (typically because WhatsApp uses the GROUP NAME as the speaker for lifecycle events).
# Conservative list — extend only when a new variant shows up in Phase 4 stats.
_SYSTEM_TEXT_PATTERNS = (
    re.compile(r"^Messages and calls are end-to-end encrypted", re.IGNORECASE),
    re.compile(r"^You created group ", re.IGNORECASE),
    re.compile(r"^You changed this group", re.IGNORECASE),
    re.compile(r"^You changed the subject", re.IGNORECASE),
    re.compile(r"^You('re| are) now an admin", re.IGNORECASE),
    re.compile(r"^You added ", re.IGNORECASE),
    re.compile(r"^You removed ", re.IGNORECASE),
    re.compile(r"^You left", re.IGNORECASE),
    re.compile(r"^You joined using this group's invite link", re.IGNORECASE),
    re.compile(r"security code (with .+ )?(has |was )?changed", re.IGNORECASE),
    re.compile(r" joined using this group's invite link$", re.IGNORECASE),
    re.compile(r" was added$", re.IGNORECASE),
    re.compile(r" left$", re.IGNORECASE),
)

_EDITED_MARKER = "<This message was edited>"


def _strip_marks(s: str) -> str:
    return _DIRECTIONAL_MARKS.sub("", s)


def _parse_timestamp(ts: str) -> datetime | None:
    for src, dst in _TIMESTAMP_SPACE_NORMALIZE.items():
        ts = ts.replace(src, dst)
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _try_parse_header(line: str) -> tuple[datetime, str] | None:
    """If `line` starts with a recognized timestamp header, return (timestamp, rest)."""
    for pattern in (_IOS_HEADER, _ANDROID_HEADER):
        m = pattern.match(line)
        if not m:
            continue
        ts = _parse_timestamp(m.group(1))
        if ts is not None:
            return ts, m.group(2)
    return None


def _is_system_text(text: str) -> bool:
    return any(p.search(text) for p in _SYSTEM_TEXT_PATTERNS)


def _finalize_message(msg: dict) -> dict | None:
    text = msg["text"].replace(_EDITED_MARKER, "").strip()
    if text in _DELETION_TEXTS:
        return None
    if _is_system_text(text):
        return None
    if _MEDIA_PATTERNS.match(text):
        text = "[media]"
    if not text:
        return None
    return {
        "timestamp": msg["timestamp"].isoformat(),
        "speaker": msg["speaker"].strip(),
        "text": text,
    }


def parse_chat_text(text: str) -> list[dict]:
    """Parse a WhatsApp `_chat.txt` body into a list of message records.

    Each record: `{"timestamp": ISO-8601 str, "speaker": str, "text": str}`.
    System messages, deleted messages, and empty messages are dropped.
    Media placeholders are normalized to `[media]`.
    """
    messages: list[dict] = []
    current: dict | None = None  # holds either a partial message or a system marker

    def flush() -> None:
        nonlocal current
        if current and current.get("kind") == "message":
            finalized = _finalize_message(current)
            if finalized is not None:
                messages.append(finalized)
        current = None

    for raw_line in text.splitlines():
        line = _strip_marks(raw_line)
        parsed = _try_parse_header(line)
        if parsed is None:
            if current and current.get("kind") == "message":
                current["text"] += "\n" + line
            continue
        flush()
        ts, rest = parsed
        speaker, sep, body = rest.partition(": ")
        if not sep:
            current = {"kind": "system"}
        else:
            current = {"kind": "message", "timestamp": ts, "speaker": speaker, "text": body}
    flush()
    return messages


def _find_chat_txt(zf: zipfile.ZipFile) -> str:
    names = zf.namelist()
    for n in names:
        if n.lower().endswith("_chat.txt"):
            return n
    for n in names:
        if n.lower().endswith(".txt"):
            return n
    raise FileNotFoundError("No .txt file found inside the zip.")


def parse_chat_zip(zip_path: Path) -> list[dict]:
    """Open a WhatsApp export zip and parse the embedded chat transcript."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        chat_member = _find_chat_txt(zf)
        with zf.open(chat_member) as f:
            raw = f.read().decode("utf-8", errors="replace")
    return parse_chat_text(raw)


def chat_name_from_zip_filename(zip_path: Path) -> str:
    """Derive a slug for the chat from the export zip's filename.

    `WhatsApp Chat - Fay.zip` → `fay`
    `WhatsApp Chat - Johan Deepak.zip` → `johan_deepak`
    `Some Other Name.zip` → `some_other_name`
    """
    stem = zip_path.stem
    prefix = "WhatsApp Chat - "
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return re.sub(r"\s+", "_", stem.strip()).lower()


def write_parsed_json(messages: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
