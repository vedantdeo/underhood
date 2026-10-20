"""Labelled figures, checked against their passage, and the conversation that teaches each answer:
the prompt, one turn of calculate calls, their results, then the figures as JSON."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from underhood import config
from underhood.tools.calc import calculate

Message = dict[str, object]

_CALL = re.compile(r"(-?\d+(?:\.\d+)?) (?:\* 10\*\*(\d+)|/ 100)")
_SCALE_WORDS = {
    3: r"thousands?|'000",
    5: r"lakhs?|lacs?",
    6: r"millions?|mn",
    7: r"crores?|cr",
    9: r"billions?|bn",
    12: r"lakh\s+crores?|trillions?",
}
_PRINTED = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ANY_SCALE = "|".join(_SCALE_WORDS.values())
# A heading that scales a table: "(Rs in crore)", "(` million)", "(USD Million)", "(Amount in ₹
# crore)": an opening bracket, at most "amount in", a currency mark and "in", then the scale word.
_CURRENCY_MARK = r"(?:Rs\.?|INR|USD|US\s?\$|EUR|GBP|[₹`$€£]|[A-Z](?![a-z]))"
_SCALE_HEADING = re.compile(
    rf"\(\s*(?:(?:amounts?|figures|value)\s+in\s+)?(?:{_CURRENCY_MARK}\s*)?(?:in\s+)?"
    rf"(?:{_CURRENCY_MARK}\s*)?(?:{_ANY_SCALE})(?![a-z])",
    re.I,
)
# aigent's chunk label gives its scale as what it multiplies by: "= ×10000000".
_MULTIPLIER = re.compile(r"= ×(\d+)\b")
_SCALE_AFTER = re.compile(rf"^\s{{0,2}}(?:{_ANY_SCALE})(?![a-z])", re.I)
_PER_UNIT_AFTER = re.compile(r"^\s{0,2}(?:per\s+(?:equity\s+)?share|each)\b", re.I)
_NUMBER_WORDS = {
    name: i
    for i, name in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split()
    )
}
_TYPOGRAPHY = str.maketrans(
    {c: "'" for c in "‘’‚‛′"} | {c: '"' for c in "“”„‟″"} | {c: "-" for c in "‐‑‒–—―"}
)
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?%)\]])")
_NEGATIVE_BEFORE = re.compile(r"(?:\(|(?<!\w)[-−–])\s*(?:[₹`A-Z]|Rs\.?)?\s*$")  # not 2023-24


@dataclass(frozen=True)
class Label:
    """One labelled figure, with the calculate call behind its value, if any."""

    metric: str
    value: float
    unit: str | None
    year: int | None
    description: str | None
    call: str | None

    def figure(self) -> dict[str, object]:
        """The figure as the answer gives it: every field but the call, in schema order."""
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "year": self.year,
            "description": self.description,
        }


def _optional(raw: Mapping[str, object], key: str, kind: type) -> object:
    value = raw.get(key)
    if value is not None and (not isinstance(value, kind) or isinstance(value, bool)):
        raise ValueError(f"{key} is not a {kind.__name__}: {value!r}")
    return value


def label(raw: Mapping[str, object]) -> Label:
    """A label from its JSON, raising ValueError for a missing field or one of the wrong type."""
    missing = {"metric", "value", "unit", "year", "call"} - raw.keys()
    if missing:
        raise ValueError(f"missing {sorted(missing)}")
    value = raw["value"]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"value is not a number: {value!r}")
    year = raw["year"]
    if isinstance(year, float) and year.is_integer():
        year = int(year)
    if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
        raise ValueError(f"year is not an integer: {year!r}")
    return Label(
        metric=str(raw["metric"]),
        value=float(value),
        unit=cast(str | None, _optional(raw, "unit", str)),
        year=year,
        description=cast(str | None, _optional(raw, "description", str)),
        call=cast(str | None, _optional(raw, "call", str)),
    )


@dataclass(frozen=True)
class Task:
    """What aigent exports about the task: the system prompt, the user turn as a template with
    {source}, {passage} and {wanted}, the tool as chat templates take it, the metrics, the ones a
    percentage may be, and the description a named metric's change carries."""

    system: str
    user: str
    tool: dict[str, object]
    metrics: tuple[str, ...]
    rate_metrics: tuple[str, ...]
    change: str

    def messages(self, passage: str, source: str, wanted: str) -> list[Message]:
        user = self.user.format(source=source, passage=passage, wanted=wanted)
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]


def load_task(path: Path) -> Task:
    """The task aigent exported."""
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    return Task(
        system=str(raw["system"]),
        user=str(raw["user"]),
        tool=cast(dict[str, object], raw["tool"]),
        metrics=tuple(cast(list[str], raw["metrics"])),
        rate_metrics=tuple(cast(list[str], raw["rate_metrics"])),
        change=str(raw["change"]),
    )


def printed(passage: str) -> set[float]:
    """Every number printed in a passage, commas dropped, plus the small ones written as words."""
    numbers = {float(n.replace(",", "")) for n in _PRINTED.findall(passage)}
    words = {float(v) for w, v in _NUMBER_WORDS.items() if re.search(rf"\b{w}\b", passage, re.I)}
    return numbers | words


def squeeze(text: str) -> str:
    """Whitespace collapsed, typography folded and case folded: how two texts compare."""
    closed = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text.translate(_TYPOGRAPHY))
    return " ".join(closed.split()).casefold()


def described(figure: Label, passage: str, change: str) -> bool:
    """A description iff the metric is other, and then a few words verbatim from the passage; on a
    named metric, none, or `change` for a change in it."""
    if figure.metric != "other":
        return figure.description in (None, change)
    if (
        not figure.description
        or len(figure.description.split()) > config.TOOL_DATA_DESCRIPTION_WORDS
    ):
        return False
    return squeeze(figure.description) in squeeze(passage)


def _named(power: int, passage: str) -> bool:
    """Whether the passage names the scale 10**power: a word for it anywhere in the text, or a
    label's multiplier."""
    if any(int(found) == 10**power for found in _MULTIPLIER.findall(passage)):
        return True
    words = _SCALE_WORDS.get(power)
    return (
        words is not None
        and re.search(rf"(?<![a-z])(?:{words})(?![a-z])", passage, re.I) is not None
    )


def _printed_negative(number: float, passage: str) -> bool:
    """Whether the passage prints `number` negative somewhere: in parentheses or after a minus."""
    for match in _PRINTED.finditer(passage):
        if float(match.group().replace(",", "")) != number:
            continue
        if _NEGATIVE_BEFORE.search(passage[max(0, match.start() - 6) : match.start()]):
            return True
    return False


def _left_unscaled(figure: Label, passage: str) -> bool:
    """Whether money given as printed belongs scaled: the passage heads it with a scale or a label's
    multiplier, or prints a scale word right after the number, and it is not a per-share figure or
    one printed per share or each."""
    if figure.unit is None or figure.value == 0 or figure.metric in config.TOOL_DATA_PER_SHARE:
        return False
    after = [
        passage[found.end() :]
        for found in _PRINTED.finditer(passage)
        if math.isclose(float(found.group().replace(",", "")), abs(figure.value))
    ]
    if any(_PER_UNIT_AFTER.match(text) for text in after):
        return False
    headed = _SCALE_HEADING.search(passage) or _MULTIPLIER.search(passage)
    return headed is not None or any(_SCALE_AFTER.match(t) for t in after)


def check_call(figure: Label, passage: str, change: str | None = None) -> bool:
    """Whether a label's value is the passage's own number, scaled by a scale the passage names or
    made a fraction by its call, the call gives that value (signed or not, for a negative), and a
    negative is printed negative, or is a `change` that is a fall. Money a passage scales, by a
    heading or a word beside it, must come through its scaling call (`_left_unscaled`)."""
    falls = change is not None and figure.description == change
    if figure.call is None:
        if figure.value < 0 and not falls and not _printed_negative(abs(figure.value), passage):
            return False
        return abs(figure.value) in printed(passage) and not _left_unscaled(figure, passage)
    match = _CALL.fullmatch(figure.call.strip())
    if match is None or (match.group(2) is not None and not _named(int(match.group(2)), passage)):
        return False
    number = abs(float(match.group(1)))
    if number not in printed(passage):
        return False
    if figure.value < 0 and not falls and not _printed_negative(number, passage):
        return False
    result = calculate(figure.call)
    if result < 0 and figure.value >= 0:
        return False
    return math.isclose(abs(result), abs(figure.value), rel_tol=1e-9, abs_tol=1e-12)


def _fits(task: Task, figure: Label) -> bool:
    """A metric the task names, and a percentage only as a rate metric, a change, or other."""
    if figure.metric not in task.metrics:
        return False
    percentage = figure.call is not None and figure.call.strip().endswith("/ 100")
    named = figure.metric != "other" and figure.metric not in task.rate_metrics
    return not (percentage and named and figure.description != task.change)


def transcript(
    task: Task, passage: str, source: str, wanted: str, labels: Sequence[Label]
) -> list[Message] | None:
    """The conversation that teaches one question, or None when any label fails `check_call`,
    is not `described`, names a metric the task does not, or is a percentage under a metric that
    cannot be one."""
    if not all(check_call(f, passage, task.change) for f in labels):
        return None
    if not all(described(f, passage, task.change) and _fits(task, f) for f in labels):
        return None
    calls = [f.call for f in labels if f.call is not None]
    messages = task.messages(passage, source, wanted)
    if calls:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "calculate", "arguments": {"expression": c}},
                    }
                    for c in calls
                ],
            }
        )
        messages.extend(
            {"role": "tool", "name": "calculate", "content": str(calculate(c))} for c in calls
        )
    figures = [f.figure() for f in labels]
    messages.append({"role": "assistant", "content": json.dumps({"figures": figures})})
    return messages
