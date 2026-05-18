"""Phase 1 — parse WhatsApp zip exports into structured JSON.

Reads:  data/raw/*.zip
Writes: data/parsed/<chat_name>.json

Run:    python scripts/01_parse_whatsapp.py
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from src.parsing import (
    chat_name_from_zip_filename,
    parse_chat_zip,
    write_parsed_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/parsed"))
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise SystemExit(f"Raw dir not found: {args.raw_dir}")

    zips = sorted(args.raw_dir.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No .zip files found in {args.raw_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_messages = 0
    for zip_path in zips:
        chat_name = chat_name_from_zip_filename(zip_path)
        out_path = args.out_dir / f"{chat_name}.json"
        try:
            messages = parse_chat_zip(zip_path)
        except Exception as e:
            print(f"  [ERROR] {zip_path.name}: {e}")
            continue
        write_parsed_json(messages, out_path)
        total_messages += len(messages)
        print(f"  {zip_path.name:50s} -> {out_path.name:30s} ({len(messages):>6,} messages)")

    print(f"\nParsed {len(zips)} chats, {total_messages:,} total messages.")


if __name__ == "__main__":
    main()
