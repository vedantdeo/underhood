"""Renting a GPU and what it costs, importable without torch.

Nothing in this package imports torch, transformers or mlx, so a caller can price or start a session
without loading them; tests/gpu/test_boundary.py holds that. A type from a heavy module goes under
`if TYPE_CHECKING:`.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"  # data.DATA_DIR, without importing numpy
