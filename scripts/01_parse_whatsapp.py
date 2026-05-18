"""Phase 1 — parse WhatsApp zip exports into structured JSON.

Reads:  data/raw/*.zip
Writes: data/parsed/<chat_name>.json

Run:    python scripts/01_parse_whatsapp.py
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/parsed"))
    args = parser.parse_args()

    raise NotImplementedError(
        "Phase 1 implementation pending. See .docs/ARCHITECTURE.md and PROJECT.md."
    )


if __name__ == "__main__":
    main()
