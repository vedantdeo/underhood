"""What a run takes on a rented GPU, in seconds and dollars, from the rates in config.

The tests in tests/gpu/test_pricing.py are the specification: estimated_seconds, rental_usd, then
hourly_usd.
"""

from __future__ import annotations

from collections.abc import Mapping


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


def hourly_usd(offer: str, rates: Mapping[str, float], tax: float) -> float:
    """What an hour of `offer` costs once `tax` is added to its listed rate in `rates`. Raises
    ValueError for an offer `rates` does not list, naming the ones it does."""
    if offer not in rates:
        raise ValueError(f"no GPU offer named {offer!r}; there is {', '.join(sorted(rates))}")
    return rates[offer] * (1 + tax)
