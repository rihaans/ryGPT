# Data schemas — ryGPT

Each phase reads one schema, writes the next. All JSON is UTF-8, indented 2 spaces for readability (JSONL is one record per line, no indentation).

## Phase 1 — parsed messages
**Path:** `data/parsed/<chat_name>.json`

```json
[
  {
    "timestamp": "2024-08-14T22:31:07",   // ISO-8601, local time, no tz (WhatsApp exports are tz-naive)
    "speaker": "Rihaan",                   // raw display name from the export
    "text": "ariyilla machaane"            // text content; multi-line messages preserved with \n
  },
  {
    "timestamp": "2024-08-14T22:31:22",
    "speaker": "Fay",
    "text": "[media]"                      // canonical placeholder; never the literal "<Media omitted>"
  }
]
```

**Filtered out at this stage:**
- System messages ("X added Y", "Messages and calls are end-to-end encrypted", "X changed the group icon", etc.)
- Empty messages
- Edited-message markers (`<This message was edited>` is stripped; the edited text itself is kept as the message)

## Phase 2 — anonymized messages
**Path:** `data/anonymized/<chat_name>.json`

Same shape as Phase 1, with substitutions:

```json
[
  {
    "timestamp": "2024-08-14T22:31:07",
    "speaker": "<self>",                   // the project owner is always <self>
    "text": "ariyilla machaane"
  },
  {
    "timestamp": "2024-08-14T22:31:22",
    "speaker": "<person_1>",               // stable per-name token
    "text": "[media]"
  }
]
```

**Name mapping (LOCAL ONLY, never committed):**
**Path:** `data/anonymized/name_mapping.json`

```json
{
  "Rihaan": "<self>",
  "Fay": "<person_1>",
  "Aaron": "<person_2>",
  "Johan Deepak": "<person_3>"
}
```

**PII scrubbed inside `text` (replaced with bracketed tokens):**
| Pattern                                       | Replacement     |
|-----------------------------------------------|-----------------|
| Phone numbers (Indian +91, generic intl, 10-digit) | `[phone]`   |
| Email addresses                               | `[email]`       |
| UPI IDs (`*@okhdfc`, `*@paytm`, `*@ybl`, `*@oksbi`, `*@axl`, `*@upi`, etc.) | `[upi]` |
| URLs containing personal paths (docs.google.com, drive.google.com with IDs) | `[link]` |
| Long digit runs (≥12 consecutive digits)      | `[number]`      |
| Generic URLs                                  | kept (configurable) |

## Phase 3 — training examples
**Path:** `data/processed/train.jsonl`, `data/processed/val.jsonl`

One JSON object per line:

```json
{
  "session_id": "fay_2024-08-14_22",         // stable session identifier; train/val split is by this
  "relationship": "gf",                       // one of: gf, friend, group
  "relationship_token": "<gf>",               // exact token to prepend at training time
  "context": [
    {"speaker": "<person_1>", "text": "evide aano?"},
    {"speaker": "<self>",      "text": "veetil aanu"},
    {"speaker": "<person_1>", "text": "ennu varum?"}
  ],
  "target": "ariyilla, oru manikoor കഴിഞ്ഞu maybe",
  "context_msg_count": 3
}
```

- `context` is up to N previous messages (default N=8, configurable).
- Last entry of `context` is the message immediately preceding `target`.
- `target` is always a `<self>` message.
- `context` may contain interleaved `<self>` messages (my own prior messages in the same session).
- `[media]` and empty `target` examples are dropped.

**Split rule:** 90/10 by `session_id` (not by example). Identical seed → identical split.

## Phase 4 — data stats
**Path:** `eval/data_stats.md` (human-readable) + `eval/data_stats.json` (machine-readable)

Stats reported:
- Total messages, my messages, per-chat breakdown
- Session count, avg/median/p95 session length (msgs)
- Token length distribution of `target` (using base model tokenizer)
- Distribution of `context_msg_count`
- Anonymization review: 20 random `target`s for manual inspection
- Group-specific stats: how often `target` immediately follows a `<self>` vs other-speaker message, distribution of speaker counts per session

## Phase 5 — tokenizer eval
**Path:** `eval/tokenizer_compression.md`

| Metric | Base tokenizer | Custom tokenizer | Ratio |
|--------|---------------:|-----------------:|------:|
| Avg tokens / message |             |                  |       |
| Total tokens for held-out 1k samples |     |          |       |
| Unique tokens hit |             |                  |       |

Decision criterion documented inline: extend if ratio ≥ 1.5x.

## Phase 7 — eval outputs
- `eval/perplexity.md` — base vs tuned, broken down per relationship
- `eval/style_classifier.md` — classifier setup, accuracy on held-out, % of tuned generations predicted as "me" + per negative-class breakdown
- `eval/samples.md` — fixed contexts + base output + tuned output, for each relationship token, 10 samples each. Manually annotated.
- `eval/memorization.md` — count and per-example listing of training targets reproduced with >80% token overlap

## Invariants enforced by tests
- Every record in any phase 1+ JSON has `timestamp`, `speaker`, `text` keys.
- No record in phase 2+ contains a real name from `name_mapping.json` (regex sweep in tests).
- Every example in phase 3 has a non-empty `target` and non-empty `context`.
- Train and val sessions are disjoint.
