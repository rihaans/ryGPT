"""WhatsApp chat export parser.

Phase 1 of the pipeline. Reads `_chat.txt` files (extracted from the WhatsApp
zip exports) and produces a list of {timestamp, speaker, text} records.

Handles:
- Both `[DD/MM/YY, HH:MM:SS]` (iOS-style) and `DD/MM/YY, HH:MM -` (Android-style) timestamps.
- Multi-line messages (continuation lines lack a leading timestamp).
- `<Media omitted>` → `[media]` placeholder.
- Edited-message markers stripped, edited content kept.
- System messages dropped.

See `.docs/DATA_SCHEMA.md` for the output schema.
"""
