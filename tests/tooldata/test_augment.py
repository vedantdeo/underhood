"""Specification for tooldata/augment.py: percentages rewritten, years shifted, companies swapped
and money rewritten, every change carried into the labels so a copy stays true, and refused
whenever it cannot."""

from __future__ import annotations

import json
import random

import pytest

from tests.tooldata.conftest import METRICS, TOOL
from underhood.tooldata import augment as figure_augment
from underhood.tooldata.augment import (
    COMPANY_FORMS,
    Answer,
    Massage,
    Swap,
    load_forms,
    load_pool,
    massage,
    massaged,
    percentages,
    rewrite_percentages,
    shift_years,
    swap_company,
    swap_currency,
    swap_rupee,
    variants,
)
from underhood.tooldata.labels import Label, Task, transcript

TASK = Task("rules", "{source}|{passage}|{wanted}", TOOL, METRICS)

PASSAGE = "The RE share rose from 30% in FY 2023-24 to 45% in FY 2024-25 across 12 plants."
QUESTION = "How did the RE share move from 30%?"
SOURCE = "Wipro Limited, Annual Report 2024-25"


def label(
    value: float,
    call: str | None,
    description: str | None = "RE share",
    unit: str | None = None,
    year: int | None = 2025,
) -> Label:
    metric = "other" if description is not None else "volume"
    return Label(metric, float(value), unit, year, description, call)


LOW = label(0.3, "30 / 100")
HIGH = label(0.45, "45 / 100")
PLANTS = label(12, None, None)


def answer(passage: str, *labels: Label, question: str = "q", source: str = SOURCE) -> Answer:
    return Answer(passage, question, source, labels)


def valid(made: Answer) -> bool:
    return transcript(TASK, made.passage, made.source, made.question, made.labels) is not None


# --- percentages -----------------------------------------------------------------------------


def test_percentages_are_rewritten_everywhere_and_stay_a_valid_transcript() -> None:
    made = rewrite_percentages(
        answer(PASSAGE, LOW, HIGH, PLANTS, question=QUESTION), random.Random(1), least=2
    )

    assert made is not None
    low, high = (float(n) for n in percentages(made.labels))
    assert {low, high}.isdisjoint({30.0, 45.0, 12.0, 2023.0, 2024.0, 2025.0}), "fresh numbers"
    assert f"from {low:g}% in FY 2023-24 to {high:g}%" in made.passage
    assert made.question == f"How did the RE share move from {low:g}%?"
    assert [m.value for m in made.labels[:2]] == pytest.approx([low / 100, high / 100])
    assert made.labels[2] == PLANTS, "a label with no percentage call is left alone"
    assert valid(made)


def test_a_percentage_in_the_description_is_rewritten_with_it() -> None:
    labels = (label(0.188, "18.8 / 100", "18.8%"), label(0.171, "17.1 / 100", None))

    made = rewrite_percentages(
        answer("Margin was 18.8% against 17.1% a year ago.", *labels), random.Random(2), 2
    )

    assert made is not None
    first = percentages(made.labels)[0]
    assert made.labels[0].description == f"{first}%" and "." in first, "one decimal kept"


@pytest.mark.parametrize(
    ("passage", "labels"),
    [
        pytest.param(PASSAGE, (LOW, PLANTS), id="fewer percentages than asked for"),
        pytest.param(
            "The share held at 30% all year.", (LOW, label(0.3, "30 / 100")), id="two share one"
        ),
        pytest.param("From 30% to 45%, then back to 30% later.", (LOW, HIGH), id="printed twice"),
        pytest.param("From 30 to 45% in the year.", (LOW, HIGH), id="not printed as a percentage"),
        pytest.param(
            "From 0% to 45% in the year.", (label(0.0, "0 / 100"), HIGH), id="no fresh number"
        ),
        pytest.param(
            "From 2% to 45% across 1 site and 3 depots.",
            (label(0.02, "2 / 100"), HIGH),
            id="every nearby number already printed",
        ),
    ],
)
def test_a_percentage_rewrite_is_refused(passage: str, labels: tuple[Label, ...]) -> None:
    assert rewrite_percentages(answer(passage, *labels), random.Random(0), least=2) is None


def test_every_seed_keeps_the_order_and_a_share_under_100() -> None:
    near = answer(
        "Occupancy fell from 90% to 80% in the year.",
        label(0.9, "90 / 100"),
        label(0.8, "80 / 100"),
    )

    for seed in range(30):
        made = rewrite_percentages(near, random.Random(seed), least=2)
        assert made is not None
        first, second = (float(n) for n in percentages(made.labels))
        assert 100 > first > second, made.passage


def test_a_longer_number_ending_in_the_same_digits_is_left_alone() -> None:
    made = rewrite_percentages(
        answer("Growth of 130% lifted the share from 30% to 45%.", LOW, HIGH), random.Random(0), 2
    )

    assert made is not None and made.passage.startswith("Growth of 130% lifted")


def test_variants_are_distinct_capped_and_the_same_every_run() -> None:
    base = answer(PASSAGE, LOW, HIGH, question=QUESTION)
    first = variants(base, copies=3, least=2, seed=0)

    assert 1 <= len(first) <= 3 and len({v.passage for v in first}) == len(first)
    assert first == variants(base, copies=3, least=2, seed=0)
    assert first != variants(base, copies=3, least=2, seed=1)


def test_variants_drop_a_repeated_draw() -> None:
    labels = (label(0.02, "2 / 100"), label(0.04, "4 / 100"))
    base = answer("From 2% to 4% across 1 site.", *labels)  # leaves 3, then 5 or 6

    made = variants(base, copies=6, least=2, seed=0)

    assert len({v.passage for v in made}) == len(made) <= 2


# --- years -----------------------------------------------------------------------------------

YEARS = "In FY 2024-25 (FY25, against CY24) revenue was 1,234, as at 31 March 2025."


def test_years_shift_in_every_form_and_in_the_labels() -> None:
    base = answer(
        YEARS,
        label(1234, None, None, year=2025),
        label(5, None, "plants", year=None),
        question="What was revenue in FY25?",
    )

    made = shift_years(base, random.Random(0), shifts=(-3,))

    assert made is not None
    assert made.passage == (
        "In FY 2021-22 (FY22, against CY21) revenue was 1,234, as at 31 March 2022."
    )
    assert made.question == "What was revenue in FY22?"
    assert made.source == "Wipro Limited, Annual Report 2021-22"
    assert [m.year for m in made.labels] == [2022, None], "a null year stays null"


@pytest.mark.parametrize(
    ("text", "shift", "expected"),
    [
        pytest.param("FY 2008-09", -10, "FY 1998-99", id="a range across the century"),
        pytest.param("FY05", -10, "FY95", id="a short year wraps"),
        pytest.param("in 2024, 2,024 units", 1, "in 2025, 2,024 units", id="a comma number"),
        pytest.param("in 2024 and 12024", 1, "in 2025 and 12024", id="a longer number"),
        pytest.param("in 2024, ratio 2024.5", 1, "in 2025, ratio 2024.5", id="a decimal"),
    ],
)
def test_year_shift_forms(text: str, shift: int, expected: str) -> None:
    made = shift_years(answer(text, label(7, None, None, year=None)), random.Random(0), (shift,))

    assert made is not None and made.passage == expected


@pytest.mark.parametrize(
    ("passage", "labelled"),
    [
        pytest.param(
            "2,050 FPOs joined in 2024.", label(2050, None, "FPOs"), id="a year-like count"
        ),
        pytest.param(
            "2024 crore in 2024", label(20240000000, "2024 * 10**7", None), id="a year-like call"
        ),
        pytest.param(
            "Twelve plants opened.", label(12, None, "plants"), id="no year in the passage"
        ),
    ],
)
def test_a_year_shift_is_refused(passage: str, labelled: Label) -> None:
    assert shift_years(answer(passage, labelled), random.Random(0), (1,)) is None


# --- companies -------------------------------------------------------------------------------

FORMS = {"Wipro Limited": ["Wipro", "Wipro Limited", "Wipro Ltd"]}
INFOSYS = Swap("Infosys Limited", "Infosys")


def test_a_company_is_renamed_by_form_and_case_everywhere() -> None:
    base = answer(
        "Wipro Limited (WIPRO) said Wipro Ltd and Wipro grew; Wiproville did not.",
        label(1, None, "Wipro offices"),
        question="How many Wipro offices?",
    )

    made = swap_company(base, random.Random(0), FORMS, [INFOSYS])

    assert made is not None
    assert made.passage == (
        "Infosys Limited (INFOSYS) said Infosys Limited and Infosys grew; Wiproville did not."
    )
    assert made.question == "How many Infosys offices?"
    assert made.source == "Infosys Limited, Annual Report 2024-25"
    assert made.labels[0].description == "Infosys offices"


@pytest.mark.parametrize(
    ("source", "pool"),
    [
        pytest.param(
            "Unknown Limited, Annual Report 2024-25", [INFOSYS], id="a company not listed"
        ),
        pytest.param(SOURCE, [], id="an empty pool"),
    ],
)
def test_a_company_swap_is_refused(source: str, pool: list[Swap]) -> None:
    base = answer("Wipro grew.", label(1, None, "offices"), source=source)

    assert swap_company(base, random.Random(0), FORMS, pool) is None


def test_the_pool_names_each_company_two_ways_and_never_a_training_one() -> None:
    pool = load_pool()
    training = {name.removesuffix(" Limited") for name in load_forms()}

    assert pool and all(swap.name.endswith(" Limited") or " of " in swap.name for swap in pool)
    assert not {swap.name.removesuffix(" Limited") for swap in pool} & training
    assert Swap("Infosys Limited", "Infosys") in pool


@pytest.mark.parametrize("key", ["companies", "pool"])
def test_the_company_file_is_sorted_by_its_key_with_no_key_twice(key: str) -> None:
    pairs = json.loads(COMPANY_FORMS.read_text(), object_pairs_hook=list)
    keys = [k for k, _ in next(v for k, v in pairs if k == key)]

    assert keys == sorted(keys) and len(keys) == len(set(keys)), key


def test_every_training_company_has_its_forms() -> None:
    forms = load_forms()

    assert len(forms) == 5
    assert all(company in written for company, written in forms.items()), "its own name"


# --- money -----------------------------------------------------------------------------------

RUPEE = label(12500000000, "1250 * 10**7", None, unit="INR")


@pytest.mark.parametrize(
    ("passage", "expected"),
    [
        pytest.param("Capex of ₹ 1,250 crore.", "Capex of Rs 1,250 crore.", id="the rupee sign"),
        pytest.param("Capex of Rs. 1,250 crore.", "Capex of Rs 1,250 crore.", id="Rs."),
        pytest.param("Capex of INR 1,250 crore.", "Capex of Rs 1,250 crore.", id="INR"),
        pytest.param("Capex of `1,250 crore.", "Capex of Rs 1,250 crore.", id="a backtick"),
        pytest.param("Capex of C 1,250 crore.", "Capex of Rs 1,250 crore.", id="a C"),
        pytest.param("Capex of I1,250 crore.", "Capex of Rs 1,250 crore.", id="an I"),
        pytest.param(
            "(₹ in crore) Capex 1,250. Phase I 2 units.",
            "(Rs in crore) Capex 1,250. Phase I 2 units.",
            id="a heading, and an I that is not a rupee",
        ),
    ],
)
def test_the_rupee_is_rewritten_and_the_label_stays_inr(
    passage: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figure_augment, "_RUPEES", ("Rs ",))

    made = swap_rupee(answer(passage, RUPEE), random.Random(0))

    assert made is not None and made.passage == expected
    assert made.labels == (RUPEE,)


@pytest.mark.parametrize(
    ("passage", "expected"),
    [
        pytest.param("Capex of ₹ 1,250 crore.", "Capex of I 1,250 crore.", id="after a space"),
        pytest.param("Capex of `1,250 crore.", "Capex of I1,250 crore.", id="attached"),
    ],
)
def test_a_symbol_keeps_the_spacing_it_found(
    passage: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figure_augment, "_RUPEES", ("I",))

    made = swap_rupee(answer(passage, RUPEE), random.Random(0))

    assert made is not None and made.passage == expected


@pytest.mark.parametrize(
    ("passage", "labelled"),
    [
        pytest.param("Capex of ₹ 1,250 crore.", label(1250, None, "plants"), id="no rupee label"),
        pytest.param("Capex of 1,250 crore.", RUPEE, id="no rupee marker"),
    ],
)
def test_a_rupee_rewrite_is_refused(passage: str, labelled: Label) -> None:
    assert swap_rupee(answer(passage, labelled), random.Random(0)) is None


MILLION = label(4500000, "4.5 * 10**6", None, unit="INR")


def test_a_currency_swap_moves_the_marker_and_every_rupee_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(figure_augment, "_FOREIGN", {"AUD": ("A$",)})
    base = answer("Revenue was ₹ 4.5 million at 12 plants (₹ in million).", MILLION, PLANTS)

    made = swap_currency(base, random.Random(0))

    assert made is not None
    assert made.passage == "Revenue was A$ 4.5 million at 12 plants (A$ in million)."
    assert [m.unit for m in made.labels] == ["AUD", None]
    assert valid(made)


@pytest.mark.parametrize(
    ("passage", "labels"),
    [
        pytest.param("Revenue was ₹ 1,250 crore.", (RUPEE,), id="counted in crore"),
        pytest.param("₹ 4.5 million, and 2 lakh more.", (MILLION,), id="a lakh elsewhere"),
        pytest.param(
            "₹ 4.5 million and USD 2 million.",
            (MILLION, label(2000000, "2 * 10**6", None, unit="USD")),
            id="another currency already",
        ),
        pytest.param("12 plants.", (PLANTS,), id="no money at all"),
        pytest.param("₹ 4.5 million; the rupee fell.", (MILLION,), id="a rupee left behind"),
        pytest.param("Revenue was 4.5 million.", (MILLION,), id="no marker to move"),
    ],
)
def test_a_currency_swap_is_refused(passage: str, labels: tuple[Label, ...]) -> None:
    assert swap_currency(answer(passage, *labels), random.Random(0)) is None


# --- massage ---------------------------------------------------------------------------------

MONEY = "In FY 2024-25 Wipro spent ₹ 4.5 million."


def how(chance: float) -> Massage:
    return Massage(chance, (-2,), FORMS, [INFOSYS])


def test_a_full_massage_moves_years_company_and_money_together() -> None:
    base = answer(MONEY, label(4500000, "4.5 * 10**6", None, unit="INR", year=2025))

    made = massage(base, random.Random(0), how(1.0))

    assert made is not None
    assert "FY 2022-23 Infosys spent" in made.passage and "₹" not in made.passage
    assert made.labels[0].year == 2023 and made.source.startswith("Infosys Limited")
    assert valid(made)


def test_no_massage_at_zero_chance() -> None:
    assert massage(answer(MONEY, MILLION), random.Random(0), how(0.0)) is None


def test_massaged_copies_are_distinct_and_the_same_every_run() -> None:
    base = answer(MONEY, MILLION)

    made = massaged(base, 3, how(1.0), seed=0)

    assert 1 <= len(made) <= 3 and len({m.passage for m in made}) == len(made)
    assert made == massaged(base, 3, how(1.0), seed=0)


def test_a_step_that_changes_nothing_does_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(figure_augment, "_RUPEES", ("₹",))
    base = answer("Capex of ₹ 1,250 crore.", RUPEE, source="Aaa Limited, Annual Report")

    assert massage(base, random.Random(0), how(1.0)) is None
