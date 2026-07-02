"""Session segmentation and training-example construction.

Phase 3. Reads anonymized JSON, produces train.jsonl / val.jsonl.

Pipeline:
1. Per chat: chronologically sort messages, split into sessions on >gap-hours.
2. For each <self> message in a session with at least one prior message: build
   a training example with the previous K messages as context.
3. Drop examples whose target is `[media]` or empty (defensive — Phase 1 already
   filters most of these).
4. Tag each example with a relationship token (`<gf>` / `<friend>` / `<group>`,
   per ADR-001 option C: collapsed relationship token, per-person speaker IDs
   preserved inside the context).
5. Split sessions 90/10 train/val (deterministic with `--seed`).

See `.docs/DATA_SCHEMA.md` for the example schema.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal

Relationship = Literal["gf", "friend", "group"]

SELF_TOKEN = "<self>"


def segment_into_sessions(
    messages: list[dict],
    gap_hours: float,
) -> list[list[dict]]:
    """Group chronologically-sorted messages into sessions on gaps > gap_hours."""
    if not messages:
        return []
    sorted_msgs = sorted(messages, key=lambda m: m["timestamp"])
    gap = timedelta(hours=gap_hours)
    sessions: list[list[dict]] = []
    current: list[dict] = []
    prev_ts: datetime | None = None
    for m in sorted_msgs:
        ts = datetime.fromisoformat(m["timestamp"])
        if prev_ts is not None and ts - prev_ts > gap:
            sessions.append(current)
            current = []
        current.append(m)
        prev_ts = ts
    if current:
        sessions.append(current)
    return sessions


def make_session_id(chat_name: str, session: list[dict]) -> str:
    """Stable session id: chat name + first-message date and hour."""
    first_ts = datetime.fromisoformat(session[0]["timestamp"])
    return f"{chat_name}_{first_ts.strftime('%Y-%m-%d_%H')}"


def build_examples_from_session(
    session: list[dict],
    session_id: str,
    relationship: Relationship,
    context_size: int,
) -> list[dict]:
    """For every <self> message with prior context, emit one training example."""
    examples: list[dict] = []
    relationship_token = f"<{relationship}>"
    for i, msg in enumerate(session):
        if msg["speaker"] != SELF_TOKEN:
            continue
        target = msg["text"]
        if not target or target == "[media]":
            continue
        start = max(0, i - context_size)
        context = session[start:i]
        if not context:
            continue
        examples.append({
            "session_id": session_id,
            "relationship": relationship,
            "relationship_token": relationship_token,
            "context": [
                {"speaker": c["speaker"], "text": c["text"]} for c in context
            ],
            "target": target,
            "context_msg_count": len(context),
        })
    return examples


def detect_relationship(
    chat_name: str,
    messages: list[dict],
    gf_chat: str,
) -> Relationship:
    """Heuristic: caller's gf_chat → 'gf'; >1 non-self speaker → 'group'; else 'friend'."""
    if chat_name == gf_chat:
        return "gf"
    non_self_speakers = {m["speaker"] for m in messages if m["speaker"] != SELF_TOKEN}
    if len(non_self_speakers) > 1:
        return "group"
    return "friend"


def build_examples_from_chat(
    chat_name: str,
    messages: list[dict],
    relationship: Relationship,
    gap_hours: float,
    context_size: int,
) -> list[dict]:
    sessions = segment_into_sessions(messages, gap_hours=gap_hours)
    out: list[dict] = []
    for session in sessions:
        sid = make_session_id(chat_name, session)
        out.extend(build_examples_from_session(
            session, sid, relationship, context_size,
        ))
    return out


def split_train_val(
    examples: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split BY session_id. Same seed → same split."""
    session_ids = sorted({ex["session_id"] for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(session_ids)
    n_val = max(1, int(round(len(session_ids) * val_fraction)))
    val_set = set(session_ids[:n_val])
    train: list[dict] = []
    val: list[dict] = []
    for ex in examples:
        (val if ex["session_id"] in val_set else train).append(ex)
    return train, val


def write_jsonl(examples: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------- Chat template formatting ----------

def example_to_chat_messages(example: dict) -> list[dict]:
    """Convert a Phase-3 example into a chat-template messages list.

    Format:
        [
          {"role": "system",    "content": "<gf>"},
          {"role": "user",      "content": "<person_3>: evide aano"},
          {"role": "assistant", "content": "<self>: veetil aanu"},
          {"role": "user",      "content": "<person_3>: ennu varum"},
          {"role": "assistant", "content": "<self>: ariyilla, oru manikoor maybe"},  # target
        ]

    - System turn carries the relationship token (`<gf>` / `<friend>` / `<group>`).
    - Consecutive same-speaker messages are merged into one turn (joined with \\n).
    - Each non-system turn is prefixed with the speaker token so the model can
      disambiguate who's talking in group chats and self-vs-other in 1:1.
    - The FINAL message is always assistant = the target. Loss is computed on
      this turn only (callers must mask everything before it).
    """
    messages: list[dict] = [{"role": "system", "content": example["relationship_token"]}]

    grouped: list[tuple[str, list[str]]] = []
    for c in example["context"]:
        if grouped and grouped[-1][0] == c["speaker"]:
            grouped[-1][1].append(c["text"])
        else:
            grouped.append((c["speaker"], [c["text"]]))

    for speaker, texts in grouped:
        role = "assistant" if speaker == SELF_TOKEN else "user"
        messages.append({"role": role, "content": f"{speaker}: " + "\n".join(texts)})

    messages.append({
        "role": "assistant",
        "content": f"{SELF_TOKEN}: {example['target']}",
    })
    return messages


def example_to_prompt_messages(example: dict) -> list[dict]:
    """Same as example_to_chat_messages but WITHOUT the final assistant turn.

    Use this for inference / eval: apply_chat_template(..., add_generation_prompt=True)
    will produce the prompt the model should continue.
    """
    return example_to_chat_messages(example)[:-1]
