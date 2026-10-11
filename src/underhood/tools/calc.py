"""Safe arithmetic for a model's `calculate` calls: numbers, + - * / ** % and parentheses."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from underhood.tools.config import MAX_EXPONENT

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression safely. No names, no calls, no attribute access. Every
    refusal, an overflow included, is a ValueError."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid expression: {expression!r}") from exc
    try:
        return _eval_node(tree.body)
    except OverflowError as exc:
        raise ValueError(f"result too large: {expression!r}") from exc


def _eval_node(node: ast.expr) -> float:
    match node:
        case ast.Constant(value=value) if isinstance(value, int | float) and not isinstance(
            value, bool
        ):
            return float(value)
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval_node(operand)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return _eval_node(operand)
        case ast.BinOp(left=left, op=op, right=right):
            fn = _BINARY_OPS.get(type(op))
            if fn is None:
                raise ValueError(f"operator not allowed: {type(op).__name__}")
            lhs, rhs = _eval_node(left), _eval_node(right)
            if isinstance(op, ast.Pow) and abs(rhs) > MAX_EXPONENT:
                raise ValueError(f"exponent too large: {rhs}")
            if isinstance(op, ast.Div | ast.Mod) and rhs == 0:
                raise ValueError("division by zero")
            return fn(lhs, rhs)
        case _:
            raise ValueError(f"unsupported syntax: {type(node).__name__}")
