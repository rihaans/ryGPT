"""Phase 4 — print data stats and sample examples for manual review.

STOP point. After this, review eval/data_stats.md and the sampled examples
for anonymization completeness before proceeding to Phase 5.

Reads:  data/processed/{train,val}.jsonl, data/anonymized/*.json
Writes: eval/data_stats.md, eval/data_stats.json

Run:    python scripts/04_data_stats.py
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--anonymized-dir", type=Path, default=Path("data/anonymized"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval"))
    parser.add_argument("--review-sample-size", type=int, default=20)
    args = parser.parse_args()

    raise NotImplementedError("Phase 4 implementation pending.")


if __name__ == "__main__":
    main()
