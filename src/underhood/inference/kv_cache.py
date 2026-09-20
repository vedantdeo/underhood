"""A KV cache, and the number that justifies it.

The tests in tests/inference/test_kv_cache.py are the specification, in this order: KVCache,
attend_step, step, generate_uncached, then generate_cached. Both generators here are greedy on
purpose, so the two must agree token for token — that is the correctness test, and it keeps this
module independent of sampling.py.

The cache is only valid while the whole sequence fits in one block: these position embeddings are
learned per absolute position, so dropping the oldest entries would silently renumber every key
left in the cache. Generate past block_size and you want RoPE, which is Week 5's problem.

main() is written for you. It times the two generators on a randomly initialised model — speed
does not depend on what the weights have learned, so the speedup is measurable before you have
trained anything.

uv run underhood-kv-bench
"""

from __future__ import annotations

import time
from collections.abc import Callable
from statistics import median

import torch
from torch import Tensor

from underhood import config
from underhood.device import pick_device
from underhood.model.attention import MultiHeadAttention
from underhood.model.gpt import GPT


class KVCache:
    """The keys and values of every position generated so far, one entry per layer.

    Keep the attribute names `keys` and `values`; the tests read them. Both start as a list of
    None, one slot per layer, and each slot grows along the time axis as positions arrive.
    """

    keys: list[Tensor | None]
    values: list[Tensor | None]

    def __init__(self, n_layers: int) -> None:
        raise NotImplementedError

    @property
    def t(self) -> int:
        """How many positions are cached — 0 before anything has been appended."""
        raise NotImplementedError

    def append(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Add one position's k and v for a layer and return that layer's full history.

        k and v: (batch, n_heads, 1, d_head). Returned: (batch, n_heads, t, d_head), the new
        position last.
        """
        raise NotImplementedError


def attend_step(attn: MultiHeadAttention, x: Tensor, cache: KVCache, layer: int) -> Tensor:
    """Attention for a single new position, against everything the cache already holds.

    x: (batch, 1, d_model) -> (batch, 1, d_model). Project x to one step of q, k and v, append the
    k and v, then attend q over the whole history. No causal mask is needed or wanted: everything
    in the cache is already in the past, which is the saving.
    """
    raise NotImplementedError


def step(model: GPT, idx_next: Tensor, cache: KVCache) -> Tensor:
    """One decode step through the model, using and extending the cache.

    idx_next: (batch, 1), the token just produced -> logits (batch, vocab) for the next one. The
    position to embed is the cache's current length, since that is how many came before.
    """
    raise NotImplementedError


def generate_uncached(model: GPT, idx: Tensor, max_new_tokens: int) -> Tensor:
    """Greedy generation the naive way: a full forward pass over the whole context every step.

    idx: (batch, t) -> (batch, t + max_new_tokens). This is the baseline the cache has to beat, so
    write it the obvious way — crop to block_size, forward, argmax the last position, append.
    """
    raise NotImplementedError


def generate_cached(model: GPT, idx: Tensor, max_new_tokens: int) -> Tensor:
    """Greedy generation through the cache: the prompt once, then one position at a time.

    Same signature and the same tokens as generate_uncached, which is how you know it is right.
    Prime the cache with the prompt before the loop; every step after it is a single position.
    """
    raise NotImplementedError


def _time(run: Callable[[], Tensor], device: torch.device, repeats: int) -> float:
    """Median seconds per call, after one warm-up that pays for MPS kernel compilation."""
    run()
    if device.type == "mps":
        torch.mps.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        run()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - started)
    return median(times)


def benchmark(
    model: GPT,
    prompt: Tensor,
    max_new_tokens: int = config.KV_BENCH_TOKENS,
    repeats: int = config.KV_BENCH_REPEATS,
) -> tuple[float, float]:
    """Median seconds for the uncached and the cached generator, in that order."""
    device = prompt.device
    with torch.no_grad():
        slow = _time(lambda: generate_uncached(model, prompt, max_new_tokens), device, repeats)
        fast = _time(lambda: generate_cached(model, prompt, max_new_tokens), device, repeats)
    return slow, fast


def main() -> None:
    torch.manual_seed(config.SEED)
    device = pick_device()
    model = GPT(block_size=config.KV_BENCH_BLOCK_SIZE).to(device).eval()
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)

    slow, fast = benchmark(model, prompt)
    tokens = config.KV_BENCH_TOKENS
    print(f"device={device}  {config.N_LAYERS} layers x {config.D_MODEL} d_model, untrained")
    print(f"{tokens} tokens, no cache:  {slow:7.2f}s  ({tokens / slow:7.1f} tok/s)")
    print(f"{tokens} tokens, KV cache:  {fast:7.2f}s  ({tokens / fast:7.1f} tok/s)")
    print(f"speedup: {slow / fast:.1f}x")


if __name__ == "__main__":
    main()
