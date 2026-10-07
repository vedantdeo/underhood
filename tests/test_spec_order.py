"""The specs run in dependency order, not alphabetically. This keeps that list honest."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import DEPENDENCY_ORDER

TESTS = Path(__file__).parent


def test_every_module_spec_appears_in_the_dependency_order() -> None:
    found = {
        f"tests/{path.relative_to(TESTS).as_posix()}"
        for path in TESTS.rglob("test_*.py")
        if path.name != Path(__file__).name
    }
    assert found == set(DEPENDENCY_ORDER), "a spec was added without a place in the build order"
