"""A task as aigent exports it, and labels, shared by the tooldata specs."""

from __future__ import annotations

import pytest

from underhood.tooldata.labels import Label, Task

METRICS = ("margin", "revenue", "profit", "debt", "market_share", "volume", "employees", "other")
RATE_METRICS = ("margin", "market_share")
TOOL: dict[str, object] = {"type": "function", "function": {"name": "calculate"}}
USER = "<source>{source}</source>\n<passage>{passage}</passage>\n<wanted>{wanted}</wanted>"


@pytest.fixture
def task() -> Task:
    return Task("rules", USER, TOOL, METRICS, RATE_METRICS, "change")


def labelled(
    value: float,
    call: str | None,
    metric: str = "revenue",
    unit: str | None = "INR",
    description: str | None = None,
    year: int | None = 2025,
) -> Label:
    return Label(metric, float(value), unit, year, description, call)
