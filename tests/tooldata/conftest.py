"""A task as aigent exports it, and labels, shared by the tooldata specs."""

from __future__ import annotations

import pytest

from underhood.tooldata.labels import Label, Task

METRICS = ("margin", "revenue", "debt", "volume", "employees", "other")
TOOL: dict[str, object] = {"type": "function", "function": {"name": "calculate"}}
USER = "<source>{source}</source>\n<passage>{passage}</passage>\n<wanted>{wanted}</wanted>"


@pytest.fixture
def task() -> Task:
    return Task(system="rules", user=USER, tool=TOOL, metrics=METRICS)


def labelled(
    value: float,
    call: str | None,
    metric: str = "revenue",
    unit: str | None = "INR",
    description: str | None = None,
    year: int | None = 2025,
) -> Label:
    return Label(metric, float(value), unit, year, description, call)
