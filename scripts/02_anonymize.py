"""Phase 2 — anonymize names and scrub PII.

Reads:  data/parsed/*.json
Writes: data/anonymized/*.json  +  data/anonymized/name_mapping.json (gitignored)

Run:    python scripts/02_anonymize.py
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", type=Path, default=Path("data/parsed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/anonymized"))
    parser.add_argument("--self-name", required=True, help="Display name for the project owner in WhatsApp (becomes <self>)")
    args = parser.parse_args()

    raise NotImplementedError("Phase 2 implementation pending.")


if __name__ == "__main__":
    main()
