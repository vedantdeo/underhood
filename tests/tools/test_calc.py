"""Specification for tools/calc.py: arithmetic in, every refusal a ValueError."""

from __future__ import annotations

import pytest

from underhood.tools.calc import calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", 3.0),
        ("(1200 * 1.07) ** 5", (1200 * 1.07) ** 5),
        ("-3 * 4", -12.0),
        ("10 % 3", 1.0),
        ("2 ** 10", 1024.0),
        ("+5 - 2", 3.0),
        ("1250 * 10**7", 12500000000.0),
    ],
)
def test_calculate(expression: str, expected: float) -> None:
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("expression", "says"),
    [
        ("__import__('os')", "unsupported syntax: Call"),
        ("a + 1", "unsupported syntax: Name"),
        ("1 / 0", "division by zero"),
        ("7 % 0", "division by zero"),
        ("2 ** 100000", "exponent too large"),
        ("(1, 2)", "unsupported syntax: Tuple"),
        ("True + 1", "unsupported syntax: Constant"),
        ("1 +", "not a valid expression"),
        ("1 << 2", "operator not allowed: LShift"),
        ("~5", "unsupported syntax: UnaryOp"),
        pytest.param("10 ** 400", "result too large", id="a float overflow under the cap"),
    ],
)
def test_calculate_rejects(expression: str, says: str) -> None:
    with pytest.raises(ValueError, match=says):
        calculate(expression)
