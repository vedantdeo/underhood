"""What a run costs before it is rented: the time arithmetic, the bill, each offer's rate, and the
card each offer rents; and what a call to a specialist hostllm serves costs."""

from __future__ import annotations

import pytest
from hostllm.pricing import TokenPrice

from underhood import config
from underhood.gpu.pricing import (
    SERVED,
    VLLM_SERVED,
    card,
    estimated_seconds,
    hourly_usd,
    rental_usd,
    served_usd,
)


@pytest.mark.parametrize(
    ("params", "tokens", "mfu", "seconds"),
    [
        pytest.param(10**6, 10**6, 1.0, 1.0, id="6 TFLOP of work at a 6 TFLOPS peak is a second"),
        pytest.param(10**6, 10**6, 0.5, 2.0, id="half the peak sustained takes twice as long"),
        pytest.param(2 * 10**6, 10**6, 1.0, 2.0, id="twice the parameters, twice the work"),
        pytest.param(10**6, 3 * 10**6, 1.0, 3.0, id="three times the tokens, three times the work"),
    ],
)
def test_estimated_seconds(params: int, tokens: int, mfu: float, seconds: float) -> None:
    assert estimated_seconds(params, tokens, 6e12, mfu) == pytest.approx(seconds)


@pytest.mark.parametrize(
    ("peak_flops", "mfu"),
    [
        pytest.param(6e12, 0.0, id="a run that sustains none of the peak"),
        pytest.param(6e12, 1.5, id="a run that beats the peak"),
        pytest.param(0.0, 0.5, id="a GPU with no peak"),
    ],
)
def test_estimated_seconds_refuses_an_impossible_gpu(peak_flops: float, mfu: float) -> None:
    with pytest.raises(ValueError):
        estimated_seconds(10**6, 10**6, peak_flops, mfu)


@pytest.mark.parametrize(
    ("seconds", "usd"),
    [
        pytest.param(3600, 1.19, id="an hour at the hourly rate"),
        pytest.param(1800, 0.595, id="half an hour, billed by the second"),
        pytest.param(0, 0.0, id="nothing run, nothing owed"),
    ],
)
def test_rental_usd(seconds: float, usd: float) -> None:
    assert rental_usd(seconds, 1.19) == pytest.approx(usd)


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "usd"),
    [
        pytest.param(1_000_000, 0, 2.0, id="a million prompt tokens at the input price"),
        pytest.param(0, 1_000_000, 30.0, id="a million answer tokens at the output price"),
        pytest.param(500, 40, 0.0022, id="a prompt and its answer, added"),
        pytest.param(0, 0, 0.0, id="nothing sent, nothing owed"),
    ],
)
def test_served_usd(input_tokens: int, output_tokens: int, usd: float) -> None:
    assert served_usd(input_tokens, output_tokens, TokenPrice(2.0, 30.0)) == pytest.approx(usd)


def test_a_served_call_is_priced_at_hostllm_s_rate_unless_told_another() -> None:
    assert served_usd(574, 36) == pytest.approx((574 * SERVED.input + 36 * SERVED.output) / 1e6)
    assert 0 < SERVED.input < SERVED.output, "a prompt token is prefilled, an answer token decoded"


def test_vllm_s_rate_is_hostllm_s_measurement_of_it() -> None:
    from hostllm.pricing import VLLM_SERVED as measured

    assert VLLM_SERVED == measured and VLLM_SERVED.output < SERVED.output


RATES = {"cheap": 1.00, "dear": 2.00}


@pytest.mark.parametrize(
    ("offer", "tax", "hourly"),
    [
        pytest.param("cheap", 0.18, 1.18, id="the listed rate with GST on top"),
        pytest.param("dear", 0.18, 2.36, id="another offer, its own rate"),
        pytest.param("cheap", 0.0, 1.00, id="no tax, the listed rate"),
    ],
)
def test_hourly_usd(offer: str, tax: float, hourly: float) -> None:
    assert hourly_usd(offer, RATES, tax) == pytest.approx(hourly)


def test_hourly_usd_names_the_offers_it_knows_when_asked_for_another() -> None:
    with pytest.raises(ValueError, match="cheap, dear"):
        hourly_usd("free", RATES, 0.18)


def test_the_default_offer_is_one_the_table_lists() -> None:
    assert config.GPU_OFFER in config.GPU_OFFERS


@pytest.mark.parametrize(
    "table",
    [
        pytest.param(config.GPU_OFFERS, id="the offers"),
        pytest.param(config.GPU_PEAK_FLOPS, id="the cards' peaks"),
        pytest.param(config.GPU_MFU, id="the cards' MFU"),
    ],
)
def test_the_tables_stay_sorted_by_name(table: dict[str, float]) -> None:
    assert list(table) == sorted(table), "one place for a new row to go"


CARDS = {"a100": 312e12, "t4": 65e12}


@pytest.mark.parametrize(
    ("offer", "expected"),
    [
        pytest.param("runpod-community-a100-80gb", "a100", id="the card sits mid-name"),
        pytest.param("kaggle-t4", "t4", id="the card ends the name"),
    ],
)
def test_card(offer: str, expected: str) -> None:
    assert card(offer, CARDS) == expected


@pytest.mark.parametrize(
    ("offer", "match"),
    [
        pytest.param("kaggle-p100", "no known card.*a100, t4", id="a card the table does not list"),
        pytest.param("kaggle-a1000", "no known card", id="a card only inside another word"),
        pytest.param("odd-a100-t4", "several", id="two cards in one offer"),
    ],
)
def test_card_refuses_an_offer_it_cannot_place(offer: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        card(offer, CARDS)


@pytest.mark.parametrize("offer", sorted(config.GPU_OFFERS))
def test_every_offer_rents_a_card_with_a_peak_and_an_mfu(offer: str) -> None:
    assert card(offer, config.GPU_PEAK_FLOPS) in config.GPU_MFU
