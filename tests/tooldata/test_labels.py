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
    exported = {"system": "rules", "user": USER, "tool": TOOL, "metrics": ["other"]}
    path.write_text(json.dumps(exported | {"rate_metrics": ["margin"], "change": "change"}))

    loaded = load_task(path)

    assert loaded.metrics == ("other",) and loaded.rate_metrics == ("margin",)
    assert loaded.change == "change"
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
        pytest.param(labelled(2871590, "2871.59 * 10**3"), False, id="thousands, never named"),
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
        pytest.param(
            labelled(1, None, "revenue", None, "change"), True, id="a named metric's change"
        ),
        pytest.param(
            labelled(1, None, "revenue", None, "growth"), False, id="change by another name"
        ),
        pytest.param(
            labelled(1, None, "other", None, "staff 1,234; (1,554.3) loss"), True, id="five words"
        ),
        pytest.param(
            labelled(1, None, "other", None, "margin 7.25%; staff 1,234; (1,554.3) loss"),
            False,
            id="six words",
        ),
    ],
)
def test_described(figure: Label, ok: bool) -> None:
    assert described(figure, TABLE, "change") is ok


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


@pytest.mark.parametrize(
    ("passage", "call", "value", "ok"),
    [
        pytest.param("Revenue (Rs in crore) 1,250", "1250 * 10**7", 12500000000, True, id="crore"),
        pytest.param(
            "Revenue 1,250 Crores", "1250 * 10**7", 12500000000, True, id="crores, any case"
        ),
        pytest.param("EBITDA K10,101 CR", "10101 * 10**7", 101010000000, True, id="CR"),
        pytest.param("(` million) 4,500", "4500 * 10**6", 4500000000, True, id="million"),
        pytest.param("USD 45 mn", "45 * 10**6", 45000000, True, id="mn"),
        pytest.param("over 3 lakh people", "3 * 10**5", 300000, True, id="lakh"),
        pytest.param("2 billion units", "2 * 10**9", 2000000000, True, id="billion"),
        pytest.param(
            "deposits of 2.5 lakh crore", "2.5 * 10**12", 2500000000000, True, id="lakh crore"
        ),
        pytest.param("(in thousands) 1,250", "1250 * 10**3", 1250000, True, id="thousands"),
        pytest.param(
            "Borrowings 110,466.1",
            "110466.1 * 10**6",
            110466100000,
            False,
            id="a scale no word names",
        ),
        pytest.param(
            "Revenue (Rs in crore) 1,250",
            "1250 * 10**6",
            1250000000,
            False,
            id="million in a crore table",
        ),
        pytest.param(
            "2.5 lakh and 3 crore", "2.5 * 10**12", 2500000000000, False, id="lakh and crore apart"
        ),
        pytest.param(
            "(` million) 4,500", "4500 * 10**7", 45000000000, False, id="crore in a million table"
        ),
        pytest.param("growth of 12%", "12 / 100", 0.12, True, id="a percentage needs no word"),
        pytest.param(
            "credit of 1,250 in the column",
            "1250 * 10**7",
            12500000000,
            False,
            id="cr and mn inside other words",
        ),
    ],
)
def test_a_scaled_call_needs_its_scale_named_in_the_passage(
    passage: str, call: str, value: float, ok: bool
) -> None:
    assert check_call(labelled(value, call), passage) is ok


@pytest.mark.parametrize(
    ("passage", "call", "value", "ok"),
    [
        pytest.param(
            "a loss of (1,554.3) in crore", "-1554.3 * 10**7", -15543000000, True, id="parentheses"
        ),
        pytest.param("finance costs (` 33) in the year", None, -33, True, id="a glyph inside them"),
        pytest.param("finance costs ` (33) in the year", None, -33, True, id="a glyph before them"),
        pytest.param("margin fell to -5 crore", "-5 * 10**7", -50000000, True, id="a minus"),
        pytest.param(
            "change of −12.5% on the year", "-12.5 / 100", -0.125, True, id="a unicode minus"
        ),
        pytest.param(
            "revenue declined by 31.75%", "-31.75 / 100", -0.3175, False, id="a fall in words"
        ),
        pytest.param("a net loss of ` 5 in the year", None, -5, False, id="a loss in words"),
        pytest.param("(5) last year and 5 this year", None, -5, True, id="printed negative once"),
        pytest.param("a profit of 5 in the year", None, 5, True, id="a positive needs nothing"),
        pytest.param(
            "a loss of (1,554.3) in crore",
            "1554.3 * 10**7",
            -15543000000,
            True,
            id="an unsigned call",
        ),
        pytest.param(
            "a gain of 1,250 crore",
            "-1250 * 10**7",
            12500000000,
            False,
            id="a signed call, positive value",
        ),
        pytest.param("margin in FY 2023-24 rose", None, -24, False, id="a year range is no minus"),
    ],
)
def test_a_negative_value_needs_the_passage_to_print_it_negative(
    passage: str, call: str | None, value: float, ok: bool
) -> None:
    figure = labelled(value, call, "profit", "INR")

    assert check_call(figure, passage) is ok


GROWTH = "Revenue grew 12% while margin was 8% and share 9%; revenue declined by 31.75% in Q1."


@pytest.mark.parametrize(
    ("figure", "taught"),
    [
        pytest.param(
            labelled(0.12, "12 / 100", "revenue", None), False, id="a percentage as revenue"
        ),
        pytest.param(
            labelled(0.12, "12 / 100", "revenue", None, "change"), True, id="revenue's change"
        ),
        pytest.param(labelled(0.08, "8 / 100", "margin", None), True, id="a margin"),
        pytest.param(labelled(0.09, "9 / 100", "market_share", None), True, id="a market share"),
        pytest.param(
            labelled(0.12, "12 / 100", "other", None, "Revenue grew"), True, id="an other"
        ),
        pytest.param(
            labelled(-0.3175, "-31.75 / 100", "revenue", None, "change"),
            True,
            id="a fall as a change",
        ),
        pytest.param(
            labelled(-0.3175, "-31.75 / 100", "profit", None), False, id="a fall in words otherwise"
        ),
    ],
)
def test_a_percentage_is_a_rate_metric_a_change_or_other(
    task: Task, figure: Label, taught: bool
) -> None:
    assert (transcript(task, GROWTH, "s", "w", [figure]) is not None) is taught
