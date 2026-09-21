"""Specification for inference/benchmark.py: it has to run and report two timings.

The interesting property — that the cache changes nothing but the speed — is a property of
generate(), so it is specified in test_sampling.py where generate lives.
"""

from __future__ import annotations

import torch

from underhood.inference.benchmark import benchmark
from underhood.model.gpt import GPT


def test_benchmark_reports_a_time_for_each_path(tiny_gpt: GPT) -> None:
    prompt = torch.zeros((1, 1), dtype=torch.long)
    slow, fast = benchmark(tiny_gpt, prompt, max_new_tokens=3, repeats=1)
    assert slow > 0 and fast > 0, f"uncached {slow}, cached {fast}"
