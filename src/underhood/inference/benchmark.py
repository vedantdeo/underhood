"""What the KV cache is worth, measured rather than assumed.

Generation itself is sampling.generate, which takes an optional cache; the two rows below are the
same function called with and without one, greedy so that the tokens are identical and only the
time differs.

A cache is only valid while the whole sequence fits one block. These position embeddings are
learned per absolute position, so dropping the oldest entries would silently renumber every key
left behind. Generate past block_size and you want RoPE, which is Week 5's problem.

The model is randomly initialised on purpose: speed does not depend on what the weights have
learned, so the number is measurable before anything is trained.

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
from underhood.inference.sampling import generate
from underhood.model.attention import KVCache
from underhood.model.gpt import GPT


def _greedy(model: GPT, prompt: Tensor, max_new_tokens: int, cached: bool) -> Tensor:
    """generate() pinned to argmax, so cached and uncached runs produce the same tokens."""
    cache = KVCache(len(model.blocks)) if cached else None
    return generate(model, prompt, max_new_tokens, top_k=1, top_p=None, cache=cache)


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
    """Median seconds without the cache and with it, in that order."""
    device = prompt.device
    slow = _time(lambda: _greedy(model, prompt, max_new_tokens, cached=False), device, repeats)
    fast = _time(lambda: _greedy(model, prompt, max_new_tokens, cached=True), device, repeats)
    return slow, fast


def main() -> None:
    torch.manual_seed(config.SEED)
    tokens = config.KV_BENCH_TOKENS
    devices = [torch.device("cpu")]
    accelerator = pick_device()
    if accelerator.type != "cpu":
        devices.append(accelerator)

    print(f"{config.N_LAYERS} layers x {config.D_MODEL} d_model, untrained, {tokens} tokens\n")
    print(f"{'device':>8}  {'no cache':>10}  {'cached':>10}  {'speedup':>8}  {'cached tok/s':>13}")
    for device in devices:
        model = GPT(block_size=config.KV_BENCH_BLOCK_SIZE).to(device).eval()
        prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
        slow, fast = benchmark(model, prompt)
        print(
            f"{device.type:>8}  {slow:>9.2f}s  {fast:>9.2f}s  {slow / fast:>7.2f}x"
            f"  {tokens / fast:>13,.0f}"
        )


if __name__ == "__main__":
    main()
