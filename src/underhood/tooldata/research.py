"""aigent's kept trajectories for v4 as SFT conversations: each with the task's tools, split by
question into train and valid. The trajectories and the task come from aigent's export.

uv run underhood-research-data --data data/research-v4
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from underhood import config

Row = dict[str, object]


def split(rows: Sequence[Row], valid_fraction: float, seed: int) -> tuple[list[Row], list[Row]]:
    """Train and valid: `valid_fraction` of the questions held out, chosen by a seeded hash of
    each question, so the file's order does not matter."""

    def rank(row: Row) -> str:
        return hashlib.sha256(f"{seed}:{row['chunk_id']}:{row['question']}".encode()).hexdigest()

    ordered = sorted(rows, key=rank)
    held = round(len(ordered) * valid_fraction)
    return ordered[held:], ordered[:held]


def build(data: Path, seed: int = config.TOOL_DATA_SEED) -> tuple[int, int]:
    """`trajectories.jsonl` and `task.json` in `data` into `train.jsonl` and `valid.jsonl` beside
    them; how many of each. ValueError for a trajectory whose system prompt is not the task's."""
    task = cast(dict[str, object], json.loads((data / "task.json").read_text(encoding="utf-8")))
    lines = (data / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [cast(Row, json.loads(line)) for line in lines if line]
    for row in rows:
        first = cast(list[dict[str, object]], row["messages"])[0]
        if first.get("content") != task["system"]:
            raise ValueError(f"{row['question']!r} was answered on another system prompt")
    train, valid = split(rows, config.TOOL_DATA_VALID_FRACTION, seed)
    for name, part in (("train", train), ("valid", valid)):
        conversations = [
            {"question": r["question"], "messages": r["messages"], "tools": task["tools"]}
            for r in part
        ]
        (data / f"{name}.jsonl").write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in conversations)
        )
    return len(train), len(valid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="aigent's export, written beside")
    args = parser.parse_args()
    train, valid = build(args.data)
    print(f"{train} train and {valid} valid conversations in {args.data}")
