"""Augmented copies of the figure fine-tune's training answers, each true by construction and
re-checked by `transcript`. Percentage-heavy answers get their percentages rewritten; any answer
may be massaged: its years shifted, its company swapped, its rupee written another way or its
currency changed. No model is called."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from underhood.tooldata.labels import Label, printed
from underhood.tools.calc import calculate

COMPANY_FORMS = Path(__file__).with_name("company-forms.json")

_PERCENT_CALL = re.compile(r"(\d+(?:\.\d+)?) / 100")
_CALLED = re.compile(r"-?\d+(?:\.\d+)?")
_TRIES = 50
_YEAR = re.compile(r"(?<!\d)(?<!\d[,.])((?:19|20)\d\d)(?:([-–/])(\d\d))?(?!\d|[,.]\d)")
_SHORT_YEAR = re.compile(r"(?<![A-Za-z])(FY|CY)(\s?'?)(\d\d)(?!\d)")
_SCALED = r"\d[\d,]*(?:\.\d+)?\s*(?:crore|lakh|million|billion|thousand)\b"
_RUPEE = re.compile(
    rf"(?<![A-Za-z])(₹|Rs\.?|INR|`)(\s?)(?=\d|in\b|crore|lakh|million|billion|thousand)"
    rf"|(?<![A-Za-z])([CI])(\s?)(?={_SCALED})"
)
_RUPEE_TRACE = re.compile(r"₹|\bRs\b|\bINR\b|`|rupee", re.I)
_INDIAN_SCALE = re.compile(r"\b(crores?|lakhs?)\b", re.I)
_RUPEES = ("₹", "Rs ", "Rs. ", "INR ", "C", "I", "`")
_FOREIGN = {
    "AUD": ("AUD ", "A$"),
    "EUR": ("EUR ", "€"),
    "GBP": ("GBP ", "£"),
    "USD": ("USD ", "US$", "$"),
}


@dataclass(frozen=True)
class Answer:
    """One labelled training answer: what `transcript` turns into a conversation."""

    passage: str
    question: str
    source: str
    labels: tuple[Label, ...]


@dataclass(frozen=True)
class Swap:
    """A company a training report's company can become: registered name and short name."""

    name: str
    short: str


Step = Callable[[Answer, random.Random], "Answer | None"]


def _everywhere(
    answer: Answer, pattern: re.Pattern[str], to: Callable[[re.Match[str]], str]
) -> Answer:
    """The answer with `pattern` rewritten in its passage, question, source and descriptions."""
    labels = tuple(
        label
        if label.description is None
        else replace(label, description=pattern.sub(to, label.description))
        for label in answer.labels
    )
    return Answer(
        pattern.sub(to, answer.passage),
        pattern.sub(to, answer.question),
        pattern.sub(to, answer.source),
        labels,
    )


def percentages(labels: Sequence[Label]) -> list[str]:
    """The printed numbers behind the labels' percentage calls, in label order."""
    found = [_PERCENT_CALL.fullmatch((label.call or "").strip()) for label in labels]
    return [match.group(1) for match in found if match is not None]


def _site(number: str) -> re.Pattern[str]:
    """Where `number` is printed as a percentage, not as part of a longer number."""
    return re.compile(rf"(?<![\d.,]){re.escape(number)}(?=\s*%)")


def _fresh(number: str, rng: random.Random, taken: set[float]) -> str | None:
    """A new number printed like `number`, within half of it either way, under 100 when it was,
    and not already in the passage; None when none turns up."""
    decimals = len(number.split(".")[1]) if "." in number else 0
    old = float(number)
    low, high = old * 0.5, old * 1.5
    if old < 100:
        high = min(high, 99.0)
    for _ in range(_TRIES):
        new = f"{rng.uniform(low, high):.{decimals}f}"
        if float(new) not in taken:
            return new
    return None


def rewrite_percentages(answer: Answer, rng: random.Random, least: int) -> Answer | None:
    """Every labelled percentage rewritten, in the same order, or None when there are fewer than
    `least`, two share a number, one is not printed exactly once, or no fresh number fits."""
    numbers = percentages(answer.labels)
    if len(numbers) < least or len({float(n) for n in numbers}) != len(numbers):
        return None
    if any(len(_site(n).findall(answer.passage)) != 1 for n in numbers):
        return None
    taken = printed(answer.passage) | printed(answer.question)
    drawn: list[str] = []
    for number in numbers:
        new = _fresh(number, rng, taken | {float(d) for d in drawn})
        if new is None:
            return None
        drawn.append(new)
    by_rank = sorted(drawn, key=float)
    order = sorted(range(len(numbers)), key=lambda i: float(numbers[i]))
    fresh = {numbers[i]: by_rank[rank] for rank, i in enumerate(order)}
    passage, question = answer.passage, answer.question
    for old, new in fresh.items():
        passage = _site(old).sub(new, passage)
        question = _site(old).sub(new, question)
    labels: list[Label] = []
    for label in answer.labels:
        match = _PERCENT_CALL.fullmatch((label.call or "").strip())
        if match is None:
            labels.append(label)
            continue
        new = fresh[match.group(1)]
        call = f"{new} / 100"
        description = label.description
        if description is not None:
            description = _site(match.group(1)).sub(new, description)
        labels.append(replace(label, call=call, value=calculate(call), description=description))
    return Answer(passage, question, answer.source, tuple(labels))


def _labelled_number(label: Label) -> float:
    """The number a label reads off the passage: its call's, or its value when it has no call."""
    if label.call is None:
        return abs(label.value)
    match = _CALLED.match(label.call.strip())
    return abs(float(match.group())) if match else abs(label.value)


def shift_years(answer: Answer, rng: random.Random, shifts: Sequence[int]) -> Answer | None:
    """Every year moved by one shift, in the text (2024, 2024-25, FY25) and in the labels, or None
    when a labelled number looks like a year or the passage names none."""
    if any(1900 <= _labelled_number(label) <= 2099 for label in answer.labels):
        return None
    if not (_YEAR.search(answer.passage) or _SHORT_YEAR.search(answer.passage)):
        return None
    shift = rng.choice(list(shifts))

    def long_year(match: re.Match[str]) -> str:
        moved = str(int(match.group(1)) + shift)
        if match.group(3) is None:
            return moved
        return f"{moved}{match.group(2)}{(int(match.group(3)) + shift) % 100:02d}"

    def short_year(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}{(int(match.group(3)) + shift) % 100:02d}"

    moved = _everywhere(_everywhere(answer, _YEAR, long_year), _SHORT_YEAR, short_year)
    labels = tuple(
        label if label.year is None else replace(label, year=label.year + shift)
        for label in moved.labels
    )
    return replace(moved, labels=labels)


def load_forms(path: Path = COMPANY_FORMS) -> dict[str, list[str]]:
    """How each training company is written: registered name -> its forms in the reports."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, list[str]], raw["companies"])


def load_pool(path: Path = COMPANY_FORMS) -> list[Swap]:
    """The companies a report may become, in ticker order."""
    raw = cast(dict[str, dict[str, str]], json.loads(path.read_text(encoding="utf-8"))["pool"])
    return [Swap(entry["name"], entry["short"]) for entry in raw.values()]


def swap_company(
    answer: Answer, rng: random.Random, forms: Mapping[str, Sequence[str]], pool: Sequence[Swap]
) -> Answer | None:
    """The report's company renamed to one from `pool` wherever it is written, or None when the
    source names a company `forms` does not list."""
    company = answer.source.split(",")[0].strip()
    if company not in forms or not pool:
        return None
    target = rng.choice(list(pool))
    written = sorted(forms[company], key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(f) for f in written) + r")(?!\w)", re.I)

    def to(match: re.Match[str]) -> str:
        found = match.group()
        new = target.name if re.search(r"\b(Limited|Ltd)$", found, re.I) else target.short
        return new.upper() if found.isupper() and len(found) > 1 else new

    return _everywhere(answer, pattern, to)


def _rupees(answer: Answer, to: Callable[[str, str], str]) -> Answer | None:
    """Every rupee marker before an amount rewritten by `to(marker, space)`; None when none is."""
    if not _RUPEE.search(answer.passage):
        return None

    def sub(match: re.Match[str]) -> str:
        marker = match.group(1) or match.group(3)
        space = match.group(2) if match.group(1) else match.group(4)
        return to(marker, space or "")

    return _everywhere(answer, _RUPEE, sub)


def swap_rupee(answer: Answer, rng: random.Random) -> Answer | None:
    """The rupee written another way the reports use (₹, Rs, INR, C, I or `), the label still INR;
    None when no label is in rupees or no marker is found."""
    if not any(label.unit == "INR" for label in answer.labels):
        return None
    glyph = rng.choice(_RUPEES)
    return _rupees(answer, lambda marker, space: glyph if glyph.endswith(" ") else glyph + space)


def swap_currency(answer: Answer, rng: random.Random) -> Answer | None:
    """Rupee amounts turned into another currency, marker and labels alike, or None when the text
    counts in lakh or crore, already has another currency, or a rupee would be left behind."""
    units = {label.unit for label in answer.labels if label.unit is not None}
    if units != {"INR"} or _INDIAN_SCALE.search(answer.passage + answer.question):
        return None
    code = rng.choice(sorted(_FOREIGN))
    glyph = rng.choice(_FOREIGN[code])
    moved = _rupees(answer, lambda marker, space: glyph if glyph.endswith(" ") else glyph + space)
    if moved is None or _RUPEE_TRACE.search(moved.passage + moved.question):
        return None
    labels = tuple(
        replace(label, unit=code) if label.unit == "INR" else label for label in moved.labels
    )
    return replace(moved, labels=labels)


@dataclass(frozen=True)
class Massage:
    """What `massage` may do to an answer, and how likely each step is."""

    chance: float
    shifts: Sequence[int]
    forms: Mapping[str, Sequence[str]]
    pool: Sequence[Swap]


def massage(answer: Answer, rng: random.Random, how: Massage) -> Answer | None:
    """The answer with each step taken at `how.chance`: years, company, then money (its currency
    changed half the time it can be, its rupee rewritten otherwise); None when nothing changed."""

    def money(a: Answer, r: random.Random) -> Answer | None:
        changed = swap_currency(a, r) if r.random() < 0.5 else None
        return changed or swap_rupee(a, r)

    steps: list[Step] = [
        lambda a, r: shift_years(a, r, how.shifts),
        lambda a, r: swap_company(a, r, how.forms, how.pool),
        money,
    ]
    out, changed = answer, False
    for step in steps:
        if rng.random() < how.chance:
            made = step(out, rng)
            if made is not None and made != out:
                out, changed = made, True
    return out if changed else None


def _distinct(made: Iterable[Answer | None], copies: int) -> list[Answer]:
    out: list[Answer] = []
    for answer in made:
        if answer is not None and all(answer.passage != a.passage for a in out):
            out.append(answer)
        if len(out) == copies:
            break
    return out


def _rng(answer: Answer, seed: int, kind: str) -> random.Random:
    return random.Random(f"{seed}:{kind}:{answer.question}:{answer.passage}")


def variants(answer: Answer, copies: int, least: int, seed: int) -> list[Answer]:
    """Up to `copies` distinct percentage rewrites, the same every run for the same seed."""
    rng = _rng(answer, seed, "percent")
    return _distinct((rewrite_percentages(answer, rng, least) for _ in range(copies)), copies)


def massaged(answer: Answer, copies: int, how: Massage, seed: int) -> list[Answer]:
    """Up to `copies` distinct massaged answers, from twice as many tries, the same every run."""
    rng = _rng(answer, seed, "massage")
    return _distinct((massage(answer, rng, how) for _ in range(copies * 2)), copies)
