"""Fixtures shared by the model, training and inference specs, and the order they run in.

The tiny GPT lives here because three test packages need one, and two copies of a fixture drift.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import pytest
import torch

from underhood.model.gpt import GPT

# Order is the data here: every spec depends on the ones above it, so `pytest -x` stops on the next
# thing to write. Do not sort this; tests/test_spec_order.py checks it stays complete.
DEPENDENCY_ORDER = (
    "tests/tokenizer/test_bpe.py",
    "tests/model/test_attention.py",
    "tests/model/test_gpt.py",
    "tests/training/test_loop.py",
    "tests/inference/test_kv_cache.py",
    "tests/inference/test_sampling.py",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Collect the specs in dependency order rather than alphabetically."""
    rank = {path: i for i, path in enumerate(DEPENDENCY_ORDER)}
    items.sort(key=lambda item: rank.get(item.location[0], len(rank)))


class TinyDims(NamedTuple):
    """Small enough that every test runs on CPU in milliseconds, real enough to be a transformer."""

    vocab_size: int = 23
    block_size: int = 8
    d_model: int = 16
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0


class GptFactory(Protocol):
    def __call__(self, **overrides: float) -> GPT: ...


@pytest.fixture
def dims() -> TinyDims:
    return TinyDims()


@pytest.fixture
def make_gpt(dims: TinyDims) -> GptFactory:
    """Build a deterministic GPT, overriding any of the tiny dimensions by name."""

    def build(**overrides: float) -> GPT:
        wanted = dims._replace(**overrides)
        torch.manual_seed(0)
        return GPT(
            vocab_size=wanted.vocab_size,
            block_size=wanted.block_size,
            d_model=wanted.d_model,
            n_heads=wanted.n_heads,
            n_layers=wanted.n_layers,
            dropout=wanted.dropout,
        ).eval()

    return build


@pytest.fixture
def tiny_gpt(make_gpt: GptFactory) -> GPT:
    return make_gpt()
