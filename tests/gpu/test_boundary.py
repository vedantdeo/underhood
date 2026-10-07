"""The light layer stays light: underhood.gpu and config import without torch, transformers or mlx.

Each module is imported in a fresh interpreter, since pytest has torch loaded long before this runs.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import pytest

import underhood.gpu

HEAVY = ("torch", "transformers", "mlx")
LIGHT = [
    "underhood",
    "underhood.config",
    "underhood.gpu",
    *(f"underhood.gpu.{info.name}" for info in pkgutil.iter_modules(underhood.gpu.__path__)),
]


@pytest.mark.parametrize("module", LIGHT)
def test_a_light_module_imports_without_the_heavy_ones(module: str) -> None:
    probe = (
        f"import importlib, sys; importlib.import_module({module!r}); "
        f"print(','.join(sorted(h for h in {HEAVY!r} if h in sys.modules)))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert loaded == "", f"importing {module} loaded {loaded}"
