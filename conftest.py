"""Make src/ and scripts/ importable when pytest runs from the repo root (CI)."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
for rel in ("src", "scripts"):
    path = str(_REPO / rel)
    if path not in sys.path:
        sys.path.insert(0, path)
