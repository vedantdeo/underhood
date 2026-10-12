"""Specification for tooldata/labels.py: labels read from JSON, checked against their passage, and
the conversation that teaches each answer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tests.tooldata.conftest import TOOL, USER, labelled
from underhood.tooldata.labels import (
    Label,
    Task,
    check_call,
    described,
    label,
    load_task,
    transcript,
)

TABLE = "Revenue (Rs in crore) 2,871.59 1,665.71; margin 7.25%; staff 1,234; (1,554.3) loss; ten"


def test_a_label_reads_its_fields_and_holds_the_value_as_a_float() -> None:
    read = label({"metric": "revenue", "value": 5, "unit": "INR", "year": 2025.0, "call": None})

    assert read == Label("revenue", 5.0, "INR", 2025, None, None)
    assert isinstance(read.value, float) and isinstance(read.year, int)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({"metric": "capex", "call": "1 * 10**7"}, id="fields missing"),
        pytest.param(
            {"metric": "x", "value": "5", "unit": None, "year": None, "call": None},
            id="a value that is not a number",
        ),
        pytest.param(
            {"metric": "x", "value": True, "unit": None, "year": None, "call": None},
            id="a boolean value",
        ),
        pytest.param(
            {"metric": "x", "value": 5, "unit": None, "year": "2025", "call": None},
            id="a year that is not an integer",
        ),
        pytest.param(
            {"metric": "x", "value": 5, "unit": 3, "year": None, "call": None},
            id="a unit that is not text",
        ),
    ],
)
def test_a_malformed_label_is_refused(raw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        label(raw)


def test_the_task_loads_and_renders_the_prompt(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps({"system": "rules", "user": USER, "tool": TOOL, "metrics": ["other"]})
    )

    loaded = load_task(path)

    assert loaded.metrics == ("other",)
    assert loaded.messages("P {x}", "S", "W") == [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "content": "<source>S</source>\n<passage>P {x}</passage>\n<wanted>W</wanted>",
        },
    ]


@pytest.mark.parametrize(
    ("figure", "ok"),
    [
        pytest.param(labelled(28715900000, "2871.59 * 10**7"), True, id="a heading's crore"),
        pytest.param(labelled(0.0725, "7.25 / 100", "margin", None), True, id="a percentage"),
        pytest.param(labelled(-15543000000, "-1554.3 * 10**7"), True, id="parentheses, negative"),
        pytest.param(labelled(1234, None, "employees", None), True, id="printed in full"),
        pytest.param(labelled(10, None, "other", None, "ten"), True, id="a number word"),
        pytest.param(labelled(2871590000, "2871.59 * 10**7"), False, id="a call it does not give"),
        pytest.param(labelled(2871590, "2871.59 * 10**3"), True, id="thousands"),
        pytest.param(labelled(28715900, "2871.59 * 10**4"), False, id="no such scale"),
        pytest.param(labelled(99990000000, "9999 * 10**7"), False, id="a number not printed"),
        pytest.param(labelled(2871.59, None), True, id="the bare cell, unscaled"),
        pytest.param(labelled(28715900000, None), False, id="scaled with no call"),
        pytest.param(labelled(28715900000, "2871.59*10**7"), False, id="a call not in the form"),
        pytest.param(labelled(5, "5 + 0"), False, id="arithmetic that is not scaling"),
    ],
)
def test_check_call(figure: Label, ok: bool) -> None:
    assert check_call(figure, TABLE) is ok


@pytest.mark.parametrize(
    ("figure", "ok"),
    [
        pytest.param(labelled(1, None, "other", None, "Staff"), True, id="verbatim, any case"),
        pytest.param(labelled(1, None, "other", None, "staff headcount"), False, id="paraphrased"),
        pytest.param(labelled(1, None, "other", None, None), False, id="other with none"),
        pytest.param(
            labelled(1, None, "revenue", None, "staff"), False, id="a named metric with one"
        ),
        pytest.param(labelled(1, None, "revenue", None, None), True, id="a named metric without"),
    ],
)
def test_described(figure: Label, ok: bool) -> None:
    assert described(figure, TABLE) is ok


def test_a_transcript_makes_every_call_in_one_turn_then_answers(task: Task) -> None:
    labels = [
        labelled(28715900000, "2871.59 * 10**7"),
        labelled(0.0725, "7.25 / 100", "margin", None),
        labelled(1234, None, "employees", None),
    ]

    messages = transcript(task, TABLE, "source", "wanted", labels)

    assert messages is not None
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    calls = cast(list[dict[str, dict[str, dict[str, str]]]], messages[2]["tool_calls"])
    assert [c["function"]["arguments"]["expression"] for c in calls] == [
        "2871.59 * 10**7",
        "7.25 / 100",
    ]
    assert [m["content"] for m in messages[3:5]] == ["28715900000.0", "0.0725"]
    final = json.loads(str(messages[-1]["content"]))["figures"]
    assert [f["value"] for f in final] == [28715900000, 0.0725, 1234]
    assert list(final[0]) == ["metric", "value", "unit", "year", "description"], "no call"


def test_a_transcript_needs_no_tool_turn_when_every_number_is_printed_in_full(task: Task) -> None:
    messages = transcript(task, TABLE, "s", "w", [labelled(1234, None, "employees", None)])

    assert messages is not None and [m["role"] for m in messages] == ["system", "user", "assistant"]


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(labelled(2871590000, "2871.59 * 10**7"), id="a call that fails its check"),
        pytest.param(labelled(1234, None, "other", None, "staff headcount"), id="paraphrased"),
        pytest.param(labelled(1234, None, "deposit", None), id="a metric the task does not name"),
    ],
)
def test_a_label_that_fails_a_check_teaches_nothing(task: Task, bad: Label) -> None:
    good = labelled(1234, None, "employees", None)

    assert transcript(task, TABLE, "s", "w", [good, bad]) is None
