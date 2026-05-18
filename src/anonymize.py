"""Anonymization and PII scrubbing.

Phase 2. Reads parsed JSON (Phase 1 output), replaces real names with stable
`<person_N>` tokens (the project owner becomes `<self>`), and scrubs PII via regex.

PII patterns scrubbed (replaced inline within message text):
- Phone numbers (Indian +91, generic international, plain 10-digit)
- Email addresses
- UPI IDs (`*@okhdfc`, `*@paytm`, `*@ybl`, `*@oksbi`, etc.)
- URLs with personal paths (Google Docs / Drive IDs)
- Long digit runs (≥12 consecutive digits — catches account / card numbers)

See `.docs/DATA_SCHEMA.md` for the output schema and the name-mapping file format.
"""
