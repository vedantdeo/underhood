"""What a run costs before it is rented: the time arithmetic, the bill, and each offer's rate."""

from __future__ import annotations

import pytest

from underhood import config
from underhood.gpu.pricing import estimated_seconds, hourly_usd, rental_usd


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


def test_the_offers_stay_sorted_by_name() -> None:
    assert list(config.GPU_OFFERS) == sorted(config.GPU_OFFERS), "one place for a new offer to go"
