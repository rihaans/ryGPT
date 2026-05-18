"""Session segmentation and training-example construction.

Phase 3. Reads anonymized JSON, produces train.jsonl / val.jsonl.

Steps:
1. Group messages into sessions on >Nh gaps (default 2h; see ADR-002 in DECISIONS.md).
2. For each `<self>` message in a session, build a training example with the
   previous K messages (default K=8) as context and the message as target.
3. Tag with relationship token (`<gf>`, `<friend>`, `<group>` — granularity per ADR-001).
4. Drop examples where target is `[media]` or empty.
5. Split by session_id (90/10) — never by message.

See `.docs/DATA_SCHEMA.md` for the training-example schema.
"""
