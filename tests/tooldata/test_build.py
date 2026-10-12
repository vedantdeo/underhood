"""Specification for tooldata/build.py: one conversation per good answer, split by chunk, with
augmented copies in train only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from underhood.tooldata import build as building
from underhood.tooldata.augment import Massage, Swap
from underhood.tooldata.build import build
from underhood.tooldata.labels import Task

STILL = Massage(0.0, (), {}, [])  # no massaged copies


def taught(value: float, call: str | None) -> dict[str, object]:
    return {
        "metric": "revenue",
        "value": value,
        "unit": "INR",
        "year": 2025,
        "description": None,
        "call": call,
    }


GOOD = taught(12500000000, "1250 * 10**7")
WRONG = taught(1250, "1250 * 10**7")
BROKEN: dict[str, object] = {"metric": "revenue", "call": "1250 * 10**7"}


def row(chunk_id: str, question: str, *figures: dict[str, object]) -> dict[str, object]:
    passage = f"Capex of Rs 1,250 crore, in {chunk_id}"
    return {
        "chunk_id": chunk_id,
        "source": "Aaa Limited",
        "passage": passage,
        "question": question,
        "figures": list(figures),
    }


def lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_build_writes_one_conversation_per_good_answer_and_drops_the_rest(
    task: Task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(building.config, "TOOL_DATA_VALID_FRACTION", 0.5)
    rows = [
        row("AAA#0001", "good", GOOD),
        row("AAA#0001", "a call that does not give the value", WRONG),
        row("AAA#0001", "no figures"),
        row("AAA#0001", "a figure missing fields", BROKEN),
        row("AAA#0002", "good too", GOOD),
    ]

    built = build(rows, task, tmp_path, seed=0, how=STILL)

    assert (built.train, built.valid, built.dropped) == (1, 1, 3)
    for name in ("train", "valid"):
        (only,) = lines(tmp_path / f"{name}.jsonl")
        assert only["tools"] == [task.tool]


def test_build_keeps_every_question_of_a_chunk_on_one_side(task: Task, tmp_path: Path) -> None:
    rows = [row(f"AAA#{i:04d}", q, GOOD) for i in range(20) for q in ("q1", "q2")]

    build(rows, task, tmp_path, seed=3, how=STILL)

    sides: dict[str, set[str]] = {}
    for name in ("train", "valid"):
        for conversation in lines(tmp_path / f"{name}.jsonl"):
            messages = conversation["messages"]
            assert isinstance(messages, list)
            sides.setdefault(str(messages[1]["content"]), set()).add(name)
    assert all(len(s) == 1 for s in sides.values())


def test_build_adds_percentage_rewrites_to_train_and_never_to_valid(
    task: Task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(building.config, "TOOL_DATA_VALID_FRACTION", 0.5)
    shares = [
        {
            "metric": "other",
            "value": v / 100,
            "unit": None,
            "year": 2025,
            "description": "RE share",
            "call": f"{v} / 100",
        }
        for v in (30, 45)
    ]
    rows = [
        {
            "chunk_id": cid,
            "source": "Aaa Limited",
            "question": "How did the RE share move?",
            "passage": f"The RE share rose from 30% to 45% at {cid}.",
            "figures": shares,
        }
        for cid in ("AAA#0001", "AAA#0002")
    ]

    built = build(rows, task, tmp_path, seed=0, how=STILL)

    assert built.valid == 1 and built.augmented >= 1 and built.massaged == 0
    assert built.train == 1 + built.augmented
    valid = (tmp_path / "valid.jsonl").read_text()
    assert "30%" in valid and "45%" in valid, "the held-out answer stays as labelled"


def test_build_adds_massaged_copies_to_train_only(
    task: Task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(building.config, "TOOL_DATA_VALID_FRACTION", 0.5)
    rows = [
        {
            "chunk_id": cid,
            "source": "Wipro Limited, Annual Report 2024-25",
            "question": "What was capex in FY 2024-25?",
            "passage": f"In FY 2024-25 Wipro spent Rs 1,250 crore at {cid}.",
            "figures": [GOOD],
        }
        for cid in ("AAA#0001", "AAA#0002")
    ]
    every = Massage(
        1.0,
        (-3,),
        {"Wipro Limited": ["Wipro", "Wipro Limited"]},
        [Swap("Infosys Limited", "Infosys")],
    )

    built = build(rows, task, tmp_path, seed=0, how=every)

    assert (built.train, built.valid, built.massaged) == (2, 1, 1)
    copy = lines(tmp_path / "train.jsonl")[1]["messages"]
    assert isinstance(copy, list)
    assert "In FY 2021-22 Infosys spent" in str(copy[1]["content"])
    assert json.loads(str(copy[-1]["content"]))["figures"][0]["year"] == 2022
    assert "Infosys" not in (tmp_path / "valid.jsonl").read_text()
