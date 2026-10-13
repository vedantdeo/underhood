"""Specification for finetune/tool_eval.py: reading tool calls, the calculator, and the loop that
answers one exported case, with the model faked."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

import pytest

from underhood.finetune.tool_eval import answer, tool_calls

Message = dict[str, object]


def call(expression: str, name: str = "calculate") -> str:
    payload = {"name": name, "arguments": {"expression": expression}}
    return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"


@pytest.mark.parametrize(
    ("text", "expressions"),
    [
        pytest.param('{"figures": []}', [], id="a direct answer"),
        pytest.param(call("2 * 10**7"), ["2 * 10**7"], id="one call"),
        pytest.param(call("1 / 100") + call("3 * 10**5"), ["1 / 100", "3 * 10**5"], id="two"),
        pytest.param("<tool_call>\n{not json}\n</tool_call>", [], id="a malformed call"),
    ],
)
def test_tool_calls(text: str, expressions: list[str]) -> None:
    got = [cast(dict[str, object], c["arguments"])["expression"] for c in tool_calls(text)]
    assert got == expressions


ROW: dict[str, object] = {
    "id": "fg-1",
    "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "q"}],
    "tools": [{"type": "function", "function": {"name": "calculate"}}],
}
FINAL = '{"figures": []}'


class Model:
    """Replies in turn, recording what it was shown each time."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.seen: list[list[Message]] = []
        self.tools: list[list[Message]] = []

    def __call__(self, messages: Sequence[Message], tools: Sequence[Message]) -> str:
        self.seen.append(list(messages))
        self.tools.append(list(tools))
        return self.replies.pop(0)


def test_a_direct_answer_is_final_with_no_calls() -> None:
    model = Model(FINAL)

    out = answer(ROW, model, rounds=3)

    assert out == {"id": "fg-1", "final": FINAL, "calls": [], "error": None}


def test_a_row_with_no_tools_is_answered_with_none_offered() -> None:
    model = Model('{"company": "Biocon"}')
    headline = {"id": "hl-001", "messages": ROW["messages"]}

    out = answer(headline, model, rounds=3)

    assert out["final"] == '{"company": "Biocon"}' and model.tools == [[]]


def test_a_call_is_run_and_its_result_shown_before_the_answer() -> None:
    model = Model(call("2 * 10**7"), FINAL)

    out = answer(ROW, model, rounds=3)

    assert out["final"] == FINAL and out["calls"] == ["2 * 10**7"] and out["error"] is None
    shown = model.seen[1]
    assert shown[2]["role"] == "assistant" and shown[2]["tool_calls"]
    assert shown[3] == {"role": "tool", "name": "calculate", "content": "20000000.0"}


@pytest.mark.parametrize(
    ("reply", "said"),
    [
        pytest.param(call("x * 2"), "Error:", id="an expression calculate refuses"),
        pytest.param(call("1 + 1", name="search"), "Error: no tool", id="a tool that is not there"),
    ],
)
def test_a_bad_call_returns_an_error_the_model_sees(reply: str, said: str) -> None:
    model = Model(reply, FINAL)

    out = answer(ROW, model, rounds=3)

    assert out["final"] == FINAL
    assert str(model.seen[1][3]["content"]).startswith(said)


def test_a_model_still_calling_after_the_last_round_has_no_answer() -> None:
    model = Model(call("1 + 1"), call("2 + 2"))

    out = answer(ROW, model, rounds=2)

    assert out["final"] is None and out["calls"] == ["1 + 1", "2 + 2"]
    assert "2 rounds" in str(out["error"])
