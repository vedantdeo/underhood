"""What a run takes on a rented GPU, in seconds and dollars, from the rates in config, and what a
call to a specialist hostllm serves costs, at the per-token price hostllm measured.

The tests in tests/gpu/test_pricing.py are the specification: estimated_seconds, rental_usd,
served_usd, hourly_usd, then card.
"""

from __future__ import annotations

from collections.abc import Mapping

from hostllm.pricing import SERVED, TokenPrice

__all__ = ["SERVED", "card", "estimated_seconds", "hourly_usd", "rental_usd", "served_usd"]


def estimated_seconds(params: int, tokens: int, peak_flops: float, mfu: float) -> float:
    """Training time from the 6 * params * tokens rule: forward and backward FLOPs over the share
    `mfu` of `peak_flops` the run sustains. Raises ValueError unless 0 < mfu <= 1 and peak > 0."""
    if not 0 < mfu <= 1:
        raise ValueError(f"mfu must be in (0, 1], got {mfu}")
    if peak_flops <= 0:
        raise ValueError(f"peak_flops must be positive, got {peak_flops}")
    return 6 * params * tokens / (peak_flops * mfu)


def rental_usd(seconds: float, usd_per_hour: float) -> float:
    """What `seconds` of a GPU rented at `usd_per_hour` costs."""
    return seconds / 3600 * usd_per_hour


def served_usd(input_tokens: int, output_tokens: int, price: TokenPrice = SERVED) -> float:
    """What a call costs at `price`, which is per million tokens: hostllm's own unless given."""
    return (input_tokens * price.input + output_tokens * price.output) / 1e6


def hourly_usd(offer: str, rates: Mapping[str, float], tax: float) -> float:
    """What an hour of `offer` costs once `tax` is added to its listed rate in `rates`. Raises
    ValueError for an offer `rates` does not list, naming the ones it does."""
    if offer not in rates:
        raise ValueError(f"no GPU offer named {offer!r}; there is {', '.join(sorted(rates))}")
    return rates[offer] * (1 + tax)


def card(offer: str, peaks: Mapping[str, float]) -> str:
    """The card `offer` rents: the one name in `peaks` that is a dash-separated part of it. Raises
    ValueError when none is, naming the cards it knows, or when several are."""
    found = [name for name in peaks if name in offer.split("-")]
    if not found:
        raise ValueError(f"no known card in {offer!r}; there is {', '.join(sorted(peaks))}")
    if len(found) > 1:
        raise ValueError(f"several cards in {offer!r}: {', '.join(sorted(found))}")
    return found[0]
