"""Specification for tooldata/research.py: aigent's kept trajectories as the conversations v4 is
trained on, each with the task's tools, split by question into train and valid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from underhood.tooldata.research import build, split

TOOLS = [{"type": "function", "function": {"name": "search_report"}}]
SYSTEM = "You answer a question about one Indian company's annual report."


def trajectory(n: int) -> dict[str, object]:
    return {
        "chunk_id": f"AAA-FY24#{n:04d}",
        "question": f"question {n}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<question>question {n}</question>"},
            {"role": "assistant", "content": '{"figures": []}'},
        ],
    }


def write(folder: Path, rows: list[dict[str, object]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "task.json").write_text(json.dumps({"system": SYSTEM, "tools": TOOLS}))


def read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_tenth_of_the_questions_are_held_out_the_same_way_every_time() -> None:
    rows = [trajectory(n) for n in range(30)]

    train, valid = split(rows, 0.1, seed=0)

    assert (len(train), len(valid)) == (27, 3)
    assert {r["question"] for r in train}.isdisjoint({r["question"] for r in valid})
    assert split(rows, 0.1, seed=0) == (train, valid) != split(rows, 0.1, seed=1)
    assert split(list(reversed(rows)), 0.1, seed=0)[1] == valid, "the file's order does not matter"


def test_each_conversation_carries_the_task_s_tools_and_its_turns_unchanged(tmp_path: Path) -> None:
    rows = [trajectory(n) for n in range(10)]
    write(tmp_path, rows)

    counts = build(tmp_path)

    train, valid = read(tmp_path / "train.jsonl"), read(tmp_path / "valid.jsonl")
    assert counts == (9, 1) == (len(train), len(valid))
    assert all(c["tools"] == TOOLS for c in [*train, *valid])
    by_question = {r["question"]: r["messages"] for r in rows}
    assert all(c["messages"] == by_question[c["question"]] for c in [*train, *valid])


def test_a_trajectory_whose_system_prompt_is_not_the_task_s_is_refused(tmp_path: Path) -> None:
    stale = trajectory(0)
    cast(list[dict[str, object]], stale["messages"])[0]["content"] = "an older prompt"
    write(tmp_path, [stale, trajectory(1)])

    with pytest.raises(ValueError, match="question 0"):
        build(tmp_path)
