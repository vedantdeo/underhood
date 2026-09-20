"""Specification for inference/kv_cache.py.

Greedy on both sides on purpose: the cached and uncached generators must produce the same tokens,
which is what makes this module testable without sampling.py existing yet.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from tests.conftest import TinyDims
from underhood.inference.kv_cache import (
    KVCache,
    attend_step,
    generate_cached,
    generate_uncached,
    step,
)
from underhood.model.attention import MultiHeadAttention
from underhood.model.gpt import GPT

Generator = Callable[[GPT, torch.Tensor, int], torch.Tensor]
BOTH: list[object] = [
    pytest.param(generate_uncached, id="uncached"),
    pytest.param(generate_cached, id="cached"),
]


def test_cache_starts_empty(dims: TinyDims) -> None:
    cache = KVCache(n_layers=dims.n_layers)
    assert cache.t == 0
    assert cache.keys == [None] * dims.n_layers
    assert cache.values == [None] * dims.n_layers


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(1, id="one position"),
        pytest.param(3, id="three positions"),
        pytest.param(8, id="a full block"),
    ],
)
def test_append_grows_the_cache_one_position_at_a_time(dims: TinyDims, steps: int) -> None:
    cache = KVCache(n_layers=1)
    d_head = dims.d_model // dims.n_heads
    appended = [torch.randn(2, dims.n_heads, 1, d_head) for _ in range(steps)]
    for k in appended[:-1]:
        cache.append(0, k, torch.randn(2, dims.n_heads, 1, d_head))
    keys, _ = cache.append(0, appended[-1], torch.randn(2, dims.n_heads, 1, d_head))
    assert cache.t == steps
    assert keys.shape == (2, dims.n_heads, steps, d_head)
    assert torch.allclose(keys[:, :, -1:], appended[-1]), "the newest position goes last"


def test_attend_step_reproduces_a_whole_causal_forward(dims: TinyDims) -> None:
    torch.manual_seed(0)
    attn = MultiHeadAttention(dims.d_model, dims.n_heads, causal=True).eval()
    x = torch.randn(2, 5, dims.d_model)
    cache = KVCache(n_layers=1)
    with torch.no_grad():
        stepped = torch.cat(
            [attend_step(attn, x[:, i : i + 1], cache, layer=0) for i in range(x.size(1))], dim=1
        )
        assert torch.allclose(attn(x), stepped, atol=1e-5)


def test_step_matches_the_models_own_last_logits(tiny_gpt: GPT, dims: TinyDims) -> None:
    idx = torch.randint(0, dims.vocab_size, (1, 4))
    cache = KVCache(n_layers=dims.n_layers)
    with torch.no_grad():
        logits, _ = tiny_gpt(idx)
        stepped = [step(tiny_gpt, idx[:, i : i + 1], cache) for i in range(idx.size(1))]
    assert stepped[-1].shape == (1, dims.vocab_size)
    assert torch.allclose(stepped[-1], logits[:, -1], atol=1e-5)


@pytest.mark.parametrize("generate_fn", BOTH)
def test_generation_keeps_the_prompt_and_adds_to_it(
    tiny_gpt: GPT, dims: TinyDims, generate_fn: Generator
) -> None:
    prompt = torch.randint(0, dims.vocab_size, (1, 3))
    out = generate_fn(tiny_gpt, prompt, 5)
    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], prompt)


def test_the_cache_changes_nothing_but_the_speed(tiny_gpt: GPT, dims: TinyDims) -> None:
    prompt = torch.randint(0, dims.vocab_size, (2, 2))
    slow = generate_uncached(tiny_gpt, prompt, 6)
    fast = generate_cached(tiny_gpt, prompt, 6)
    assert torch.equal(slow, fast), f"cached {fast.tolist()} != uncached {slow.tolist()}"
