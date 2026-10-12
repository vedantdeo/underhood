"""Labelled answers as tool-use SFT conversations, split by chunk, with augmented copies of each
training answer. The answers and the task come from aigent's export.

uv run underhood-tool-data --answers data/figures-v3/answers.jsonl \\
    --task data/figures-v3/task.json --out data/figures-v3
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from underhood import config
from underhood.tooldata.augment import Answer, Massage, load_forms, load_pool, massaged, variants
from underhood.tooldata.labels import Label, Message, Task, label, load_task, transcript


@dataclass(frozen=True)
class Built:
    """What `build` wrote and what it dropped."""

    train: int
    valid: int
    dropped: int
    augmented: int
    massaged: int


def massage_settings() -> Massage:
    """Config's massage, over the shipped company forms and pool."""
    return Massage(
        config.TOOL_DATA_MASSAGE_CHANCE,
        config.TOOL_DATA_MASSAGE_SHIFTS,
        load_forms(),
        load_pool(),
    )


def build(
    rows: Sequence[Mapping[str, object]],
    task: Task,
    out_dir: Path,
    seed: int = config.TOOL_DATA_SEED,
    how: Massage | None = None,
) -> Built:
    """Every answered question as a conversation, split by chunk so a passage never sits on both
    sides, plus percentage rewrites and massaged copies (`how`, config's by default) of each
    training answer. An answer with no figures, a malformed one or one `transcript` refuses is
    dropped; a copy `transcript` refuses is skipped."""
    how = how or massage_settings()
    chunk_ids = sorted({str(row["chunk_id"]) for row in rows})
    held_out = round(len(chunk_ids) * config.TOOL_DATA_VALID_FRACTION)
    held = set(random.Random(seed).sample(chunk_ids, held_out))
    splits: dict[str, list[list[Message]]] = {"train": [], "valid": []}
    dropped = augmented = massaged_count = 0
    for row in rows:
        messages: list[Message] | None = None
        figures: list[Label] = []
        try:
            figures = [label(f) for f in cast(list[Mapping[str, object]], row["figures"])]
            if figures:
                passage, source = str(row["passage"]), str(row["source"])
                messages = transcript(task, passage, source, str(row["question"]), figures)
        except ValueError:
            messages = None
        if not messages:
            dropped += 1
            continue
        if row["chunk_id"] in held:
            splits["valid"].append(messages)
            continue
        splits["train"].append(messages)
        answer = Answer(
            str(row["passage"]), str(row["question"]), str(row["source"]), tuple(figures)
        )
        for kind, copies in (
            (
                "percent",
                variants(
                    answer, config.TOOL_DATA_AUGMENT_COPIES, config.TOOL_DATA_AUGMENT_LEAST, seed
                ),
            ),
            ("massage", massaged(answer, config.TOOL_DATA_MASSAGE_COPIES, how, seed)),
        ):
            for copy in copies:
                rewritten = transcript(task, copy.passage, copy.source, copy.question, copy.labels)
                if not rewritten:
                    continue
                splits["train"].append(rewritten)
                if kind == "percent":
                    augmented += 1
                else:
                    massaged_count += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, conversations in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for messages in conversations:
                out.write(json.dumps({"messages": messages, "tools": [task.tool]}) + "\n")
    return Built(len(splits["train"]), len(splits["valid"]), dropped, augmented, massaged_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tool-use SFT data from labelled answers.")
    parser.add_argument("--answers", type=Path, required=True, help="aigent's answers.jsonl")
    parser.add_argument("--task", type=Path, required=True, help="aigent's task.json")
    parser.add_argument("--out", type=Path, required=True, help="where train/valid go")
    args = parser.parse_args()
    lines = args.answers.read_text(encoding="utf-8").splitlines()
    rows = [cast(dict[str, object], json.loads(x)) for x in lines if x.strip()]
    built = build(rows, load_task(args.task), args.out)
    print(
        f"{built.train} train ({built.augmented} percentage rewrites, {built.massaged} massaged), "
        f"{built.valid} valid, {built.dropped} dropped -> {args.out}"
    )


if __name__ == "__main__":
    main()
