"""Phase 8 (optional) — local Gradio demo.

GATED: do not run if Phase 7 memorization audit failed (>5% flagged).
Local-only by default; do not bind to 0.0.0.0 or share=True without an explicit decision.

Reads:  models/lora_adapter/
Serves: http://127.0.0.1:7860

Run:    python scripts/08_demo.py
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/lora_adapter"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    raise NotImplementedError("Phase 8 implementation pending.")


if __name__ == "__main__":
    main()
