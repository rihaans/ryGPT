"""Phase 2 — anonymize names and scrub PII.

Reads:  data/parsed/*.json
Writes: data/anonymized/*.json  +  data/anonymized/name_mapping.json (gitignored)

Run:    python scripts/02_anonymize.py

On first run, the name mapping is generated automatically from speaker fields.
The mapping is preserved across re-runs — edit `data/anonymized/name_mapping.json`
to add nickname aliases (e.g. ["fay", "Faiu"] all mapped to <person_N>) and re-run.
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from src.anonymize import (
    anonymize_messages,
    build_mapping_from_speakers,
    collect_speakers,
    load_mapping,
    save_mapping,
    write_anonymized_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", type=Path, default=Path("data/parsed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/anonymized"))
    parser.add_argument(
        "--self-name",
        default="rihaan",
        help="WhatsApp display name of the project owner (becomes <self>)",
    )
    parser.add_argument(
        "--scrub-urls",
        action="store_true",
        help="If set, ALL URLs (not just Google Docs/Drive) become [link].",
    )
    args = parser.parse_args()

    if not args.in_dir.exists():
        raise SystemExit(f"Input dir not found: {args.in_dir}. Run Phase 1 first.")

    parsed_files = sorted(args.in_dir.glob("*.json"))
    if not parsed_files:
        raise SystemExit(f"No parsed JSON files found in {args.in_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = args.out_dir / "name_mapping.json"

    # Build or extend the mapping.
    speakers = collect_speakers(parsed_files)
    existing = load_mapping(mapping_path)
    mapping = build_mapping_from_speakers(
        speakers, self_name=args.self_name, existing=existing
    )
    save_mapping(mapping, mapping_path)
    print(f"Name mapping: {len(mapping)} tokens, saved to {mapping_path}")
    for token, aliases in mapping.items():
        print(f"  {token:14s} -> {aliases}")
    print()

    # Anonymize each chat.
    total_in = total_out = 0
    for p in parsed_files:
        with p.open(encoding="utf-8") as f:
            messages = json.load(f)
        anonymized = anonymize_messages(
            messages, mapping, keep_urls=not args.scrub_urls
        )
        out_path = args.out_dir / p.name
        write_anonymized_json(anonymized, out_path)
        total_in += len(messages)
        total_out += len(anonymized)
        print(f"  {p.name:30s} -> {out_path.name:30s} ({len(anonymized):>6,} messages)")

    print(f"\nAnonymized {len(parsed_files)} chats, {total_out:,} messages.")


if __name__ == "__main__":
    main()
