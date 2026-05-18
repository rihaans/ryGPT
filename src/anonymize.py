"""Anonymization and PII scrubbing.

Phase 2. Reads parsed JSON (Phase 1 output), replaces real names with stable
`<person_N>` tokens (project owner becomes `<self>`), and scrubs PII via regex.

Name mapping format (data/anonymized/name_mapping.json — LOCAL ONLY):
    {
      "<self>": ["rihaan"],
      "<person_1>": ["Aaron"],
      "<person_2>": ["fay", "Faiu"],     # multiple aliases (nicknames) ok
      ...
    }
The mapping is persisted across runs. New speakers detected in subsequent
runs get fresh `<person_N>` tokens appended; existing entries are preserved.
Users can edit the JSON to add nickname aliases or merge entries.

PII patterns (applied to message text in this order):
  UPI ids       → [upi]      (e.g. name@paytm, foo@okhdfc)
  Emails        → [email]
  Intl phones   → [phone]    (e.g. +91 87788 90419, +1-555-123-4567)
  Indian mobile → [phone]    (10 digits starting 6-9, optionally 5+5 split)
  Long digits   → [number]   (≥12 consecutive digits — account/card numbers)
  Google links  → [link]     (docs/drive personal-identifier URLs)
  Generic URLs  → kept by default; pass keep_urls=False to scrub

See `.docs/DATA_SCHEMA.md` for the output schema.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

# ---------- PII patterns ----------

# UPI handle: `name@<provider>` where provider is one of the common Indian PSPs.
_UPI_PROVIDERS = (
    "okhdfc", "okhdfcbank", "okaxis", "okicici", "oksbi",
    "paytm", "ybl", "axl", "upi", "ibl", "apl",
    "axisbank", "hdfcbank", "sbi", "federal", "kotak", "icici",
)
_UPI_RE = re.compile(
    r"\b[A-Za-z0-9._-]+@(?:" + "|".join(_UPI_PROVIDERS) + r")\b",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Separators commonly seen inside a phone number: any whitespace, NBSP ( ),
# narrow NBSP ( ), or hyphen.
_PHONE_SEP = "[\\s  -]"

# International phone: '+' then 7-15 digits, with optional separators interleaved.
# Pattern: 6-14 (digit + optional separator) followed by a final digit ⇒ total 7-15 digits.
_PHONE_INTL_RE = re.compile(
    r"(?<!\d)\+(?:\d" + _PHONE_SEP + r"?){6,14}\d(?!\d)"
)

# Indian mobile: 10 digits starting 6-9, optionally split 5+5 with a single separator
# (matches WhatsApp contact-display style "98765 43210" with narrow NBSP).
_PHONE_10_RE = re.compile(
    r"(?<!\d)[6-9]\d{4}" + _PHONE_SEP + r"?\d{5}(?!\d)"
)

# Suspiciously long digit runs (card/account numbers).
_LONG_DIGIT_RE = re.compile(r"(?<!\d)\d{12,}(?!\d)")

# Google Docs/Drive — these encode personal document IDs.
_GOOGLE_LINK_RE = re.compile(r"https?://(?:docs|drive)\.google\.com/\S+", re.IGNORECASE)

# Catch-all URL.
_GENERIC_URL_RE = re.compile(r"https?://\S+")

# Order: UPI first (so paytm/ybl handles don't get caught by partial digit patterns later);
# email next; phones; long digits; google links last among always-scrub patterns.
_SCRUB_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (_UPI_RE, "[upi]"),
    (_EMAIL_RE, "[email]"),
    (_PHONE_INTL_RE, "[phone]"),
    (_PHONE_10_RE, "[phone]"),
    (_LONG_DIGIT_RE, "[number]"),
    (_GOOGLE_LINK_RE, "[link]"),
)


def scrub_text(text: str, keep_urls: bool = True) -> str:
    """Apply PII regex scrub to a string. URLs are kept unless `keep_urls=False`."""
    for pattern, replacement in _SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    if not keep_urls:
        text = _GENERIC_URL_RE.sub("[link]", text)
    return text


# ---------- Name mapping ----------

_PERSON_TOKEN_RE = re.compile(r"^<person_(\d+)>$")


def build_mapping_from_speakers(
    speakers: Iterable[str],
    self_name: str,
    existing: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Build or extend a token → aliases mapping.

    Existing entries are preserved verbatim. New speakers (not present in any
    existing alias list) get freshly assigned `<person_N>` tokens, with N
    incremented past the largest existing person index. New speakers are
    assigned in case-insensitive alphabetical order for stability.
    """
    mapping: dict[str, list[str]] = {k: list(v) for k, v in (existing or {}).items()}

    # Ensure <self> entry exists and contains the canonical self_name.
    self_aliases = mapping.setdefault("<self>", [])
    if self_name not in self_aliases:
        self_aliases.append(self_name)

    # Flatten: lowercased alias → token (for membership check).
    known: dict[str, str] = {
        a.lower(): tok for tok, aliases in mapping.items() for a in aliases
    }

    # Largest existing person index, so we keep numbering monotonic.
    max_idx = 0
    for tok in mapping:
        m = _PERSON_TOKEN_RE.match(tok)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    for sp in sorted(set(speakers), key=str.lower):
        if sp.lower() == self_name.lower():
            continue
        if sp.lower() in known:
            continue
        max_idx += 1
        token = f"<person_{max_idx}>"
        mapping[token] = [sp]
        known[sp.lower()] = token

    return mapping


def _build_replacement_regex(
    mapping: dict[str, list[str]],
) -> tuple[re.Pattern | None, dict[str, str]]:
    alias_to_token: dict[str, str] = {}
    for token, aliases in mapping.items():
        for alias in aliases:
            alias_to_token[alias.lower()] = token
    if not alias_to_token:
        return None, {}
    # Longest aliases first so multi-word names win over single-word substrings.
    ordered = sorted(alias_to_token, key=lambda a: (-len(a), a))
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(a) for a in ordered) + r")\b",
        re.IGNORECASE,
    )
    return pattern, alias_to_token


def replace_names(
    text: str,
    name_regex: re.Pattern | None,
    alias_to_token: dict[str, str],
) -> str:
    if name_regex is None:
        return text

    def _sub(m: re.Match) -> str:
        return alias_to_token.get(m.group(0).lower(), m.group(0))

    return name_regex.sub(_sub, text)


# ---------- Message anonymization ----------

def anonymize_messages(
    messages: list[dict],
    mapping: dict[str, list[str]],
    keep_urls: bool = True,
) -> list[dict]:
    """Apply mapping + scrub to every message. Unknown speakers passed through unchanged."""
    name_regex, alias_to_token = _build_replacement_regex(mapping)
    speaker_to_token: dict[str, str] = {
        a.lower(): tok for tok, aliases in mapping.items() for a in aliases
    }

    out: list[dict] = []
    for msg in messages:
        speaker = speaker_to_token.get(msg["speaker"].lower(), msg["speaker"])
        text = scrub_text(msg["text"], keep_urls=keep_urls)
        text = replace_names(text, name_regex, alias_to_token)
        out.append({
            "timestamp": msg["timestamp"],
            "speaker": speaker,
            "text": text,
        })
    return out


# ---------- IO ----------

def load_mapping(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_mapping(mapping: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def write_anonymized_json(messages: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def collect_speakers(parsed_files: Iterable[Path]) -> list[str]:
    """Gather the set of all speaker strings across a list of parsed JSON files."""
    seen: set[str] = set()
    for p in parsed_files:
        with p.open(encoding="utf-8") as f:
            for msg in json.load(f):
                seen.add(msg["speaker"])
    return sorted(seen, key=str.lower)
