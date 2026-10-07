"""Specification for inference/benchmark.py: it has to run and report two timings.

The interesting property — that the cache changes nothing but the speed — is a property of
generate(), so it is specified in test_sampling.py where generate lives.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

import underhood.inference.benchmark as bench
from underhood.inference.benchmark import benchmark
from underhood.model.gpt import GPT


def test_benchmark_reports_a_time_for_each_path(tiny_gpt: GPT) -> None:
    prompt = torch.zeros((1, 1), dtype=torch.long)
    slow, fast = benchmark(tiny_gpt, prompt, max_new_tokens=3, repeats=1)
    assert slow > 0 and fast > 0, f"uncached {slow}, cached {fast}"


def test_throughput_counts_every_sequence_in_the_batch(
    tiny_gpt: GPT, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[tuple[int, ...], int, bool]] = []

    def greedy(model: GPT, prompt: Tensor, steps: int, cached: bool) -> Tensor:
        sent.append((tuple(prompt.shape), steps, cached))
        return prompt

    def timed(run: object, device: torch.device, repeats: int) -> float:
        assert callable(run)
        run()
        return 0.5

    monkeypatch.setattr(bench, "_greedy", greedy)
    monkeypatch.setattr(bench, "_time", timed)

    rate = bench.throughput(tiny_gpt, torch.device("cpu"), batch=16, max_new_tokens=8)

    assert rate == 16 * 8 / 0.5
    assert sent == [((16, 1), 8, True)], "one prompt token a row, with the cache"
