"""A sandbox with one file in it, and a file just outside it to escape to."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def box(tmp_path: Path) -> Path:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "notes.txt").write_text("hello from the sandbox\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("secret\n", encoding="utf-8")
    return sandbox
