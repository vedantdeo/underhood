"""Fetch the small corpora used for training from scratch.

uv run underhood-fetch
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def tiny_shakespeare() -> Path:
    """Download tiny Shakespeare (about 1 MB) once; return its path."""
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / "tinyshakespeare.txt"
    if not target.exists():
        print(f"downloading {TINY_SHAKESPEARE}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE, target)
    return target


def main() -> None:
    path = tiny_shakespeare()
    print(f"{path}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
