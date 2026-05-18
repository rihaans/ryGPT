"""Add the project root to sys.path so scripts can `from src import ...`.

Imported (for side effect) as the first line of every `scripts/0X_*.py`.
Kept as a separate module so the path-manipulation idiom isn't duplicated.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
